"""
Feature Prior Explainer Module

Analyzes the image feature prior F that combines Transformer (global) and
CNN (local) features to condition the diffusion denoising process.

Features:
- Extracts raw features from Transformer encoder (full image)
- Extracts ROI features from CNN encoder (local patches)
- Analyzes learnable fusion weights Q
- Computes feature contribution scores
- Tracks feature magnitudes and norms
- Provides ROI importance rankings

Purpose:
- Understand how different feature sources contribute to predictions
- Analyze feature fusion mechanisms
- Identify which ROIs are most important
- Support feature engineering and model optimization
- Enable research on multi-modal feature fusion

Future Extensions:
- Feature prior evolution through diffusion timesteps
- Attention-weighted feature analysis
- Feature clustering and similarity analysis
- Counterfactual feature analysis
- Feature ablation studies
- Multi-scale feature pyramid visualization
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from xai.core.base_explainer import BaseExplainer


class FeaturePriorExplainer(BaseExplainer):
    """
    Analyze feature prior composition from image and ROI features.
    
    The feature prior F combines:
    - F_raw: Transformer encoder output (full image)
    - F_rois: CNN encoder outputs (local patches/ROIs)
    - Q: Learnable fusion weights
    
    Attributes:
        model: CoolSystem containing conditional model
        cond_model: The ConditionalModel (U-Net)
        config: Configuration dictionary
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device, config: Dict[str, Any]):
        """
        Initialize the feature prior explainer.
        
        Args:
            model: The CoolSystem model
            device: Device to run on
            config: Configuration with settings
        """
        super().__init__(model, device, config)
        
        self.cond_model = model.model
        self.aux_model = model.aux_model
        self.cond_model.eval()
        self.aux_model.eval()
        
        self.num_classes = config.get('num_classes', 5)
        self.model_config = model.params
        
        print(f"[FeaturePriorExplainer] Initialized")
        print(f"  Number of classes: {self.num_classes}")
        print(f"  Device: {self.device}")
    
    def explain(self,
                image: torch.Tensor,
                label: Optional[int] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Generate feature prior explanation.
        
        Args:
            image: Input image tensor of shape (C, H, W) or (1, C, H, W)
            label: Ground truth label (optional)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing:
            - 'explanation_type': 'feature_prior'
            - 'prediction': Predicted class
            - 'confidence': Confidence score
            - 'raw_features': Transformer encoder output
            - 'roi_features': CNN encoder outputs for patches
            - 'fusion_weights': Learnable fusion weights Q
            - 'fused_features': Final fused feature prior
            - 'contribution_scores': Contribution of each feature source
        """
        image = self._preprocess_image(image)
        
        # Get patches from aux model
        with torch.no_grad():
            (y_fusion, y_global, y_local,
             patches, patch_attns, saliency_map) = self.aux_model(image)
        
        # Extract features using conditional model encoders
        # Note: We need to hook into the model to get intermediate features
        raw_features, roi_features, fusion_weights = self._extract_features(
            image, patches, patch_attns
        )
        
        # Compute fused features
        fused_features = self._fuse_features(raw_features, roi_features, fusion_weights)
        
        # Compute contribution scores
        contribution_scores = self._compute_contributions(
            raw_features, roi_features, fusion_weights
        )
        
        # Get predictions
        probs_fusion = F.softmax(y_fusion, dim=1)[0]
        prediction = int(torch.argmax(probs_fusion, dim=0).item())
        confidence = float(probs_fusion[prediction].item())
        
        explanation = self.get_explanation_dict(
            explanation_type='feature_prior',
            prediction=prediction,
            confidence=confidence,
            
            # Feature data
            raw_features=raw_features.detach().cpu().numpy(),
            roi_features=roi_features.detach().cpu().numpy(),
            fusion_weights=fusion_weights.detach().cpu().numpy(),
            fused_features=fused_features.detach().cpu().numpy(),
            contribution_scores=contribution_scores,
            
            # Feature statistics
            raw_feature_norm=np.linalg.norm(raw_features.detach().cpu().numpy()),
            roi_feature_norm=np.linalg.norm(roi_features.detach().cpu().numpy()),
            fusion_weight_norm=np.linalg.norm(fusion_weights.detach().cpu().numpy()),
            
            # Patch information
            num_patches=patches.shape[1] if patches is not None else 0,
            patch_attention=patch_attns[0].detach().cpu().numpy() if patch_attns is not None else None,
            
            # Ground truth
            ground_truth=label if label is not None else -1
        )
        
        return explanation
    
    def _extract_features(self,
                         image: torch.Tensor,
                         patches: torch.Tensor,
                         patch_attns: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract raw and ROI features from encoders.
        
        Args:
            image: Input image [B, C, H, W]
            patches: Patch crops [B, N_p, I, J]
            patch_attns: Patch attention weights [B, N_p]
            
        Returns:
            Tuple of (raw_features, roi_features, fusion_weights)
        """
        # Get encoders from conditional model
        encoder_x = self.cond_model.encoder_x
        encoder_x_l = self.cond_model.encoder_x_l
        
        # Extract raw features (Transformer encoder)
        with torch.no_grad():
            raw_features = encoder_x(image)  # [B, feature_dim, H', W']
            raw_features = self.cond_model.norm(raw_features)
        
        # Extract ROI features (CNN encoder)
        bz, np, I, J = patches.shape
        
        # Reshape patches for encoder
        patches_reshaped = patches.view(bz * np, I, J).unsqueeze(1).expand(-1, 3, -1, -1)
        
        with torch.no_grad():
            roi_features = encoder_x_l(patches_reshaped)  # [B*N_p, feature_dim, H'', W'']
            roi_features = self.cond_model.norm_l(roi_features)
        
        # Reshape ROI features
        roi_features = roi_features.view(bz, np, roi_features.shape[1])  # [B, N_p, feature_dim]
        
        # Get fusion weights
        fusion_weights = self.cond_model.cond_weight  # [1, feature_dim, 7]
        fusion_weights = F.softmax(fusion_weights, dim=2)  # Normalize
        
        return raw_features, roi_features, fusion_weights
    
    def _fuse_features(self,
                      raw_features: torch.Tensor,
                      roi_features: torch.Tensor,
                      fusion_weights: torch.Tensor) -> torch.Tensor:
        """
        Fuse raw and ROI features using learnable weights.
        
        This replicates the fusion process in ConditionalModel.forward().
        
        Args:
            raw_features: Raw features [B, feature_dim, H, W]
            roi_features: ROI features [B, N_p, feature_dim]
            fusion_weights: Fusion weights [1, feature_dim, K+1] where K+1 = N_p+1
            
        Returns:
            Fused features [B, feature_dim, H, W]
        """
        bz = raw_features.shape[0]
        feature_dim = raw_features.shape[1]
        
        # Prepare raw features: add spatial dimension for concatenation
        # In the model: x = torch.cat([x.unsqueeze(-1), x_l], dim=-1)
        # x is [B, feature_dim, H, W] -> unsqueeze to [B, feature_dim, H, W, 1]
        # x_l is [B, N_p, feature_dim] -> reshape and permute
        
        # For visualization, we'll compute a simplified fusion
        # that shows how features are combined
        
        # Average pool raw features to get global representation
        raw_global = F.adaptive_avg_pool2d(raw_features, (1, 1)).squeeze(-1).squeeze(-1)  # [B, feature_dim]
        
        # Average ROI features
        roi_avg = roi_features.mean(dim=1)  # [B, feature_dim]
        
        # Combine using fusion weights (simplified)
        # In actual model, this happens spatially, but for explanation we use averages
        fused = (raw_global + roi_avg) / 2  # Simplified fusion
        
        # Expand back to spatial dimensions
        fused = fused.unsqueeze(-1).unsqueeze(-1).expand_as(raw_features)
        
        return fused
    
    def _compute_contributions(self,
                              raw_features: torch.Tensor,
                              roi_features: torch.Tensor,
                              fusion_weights: torch.Tensor) -> Dict[str, float]:
        """
        Compute contribution scores for each feature source.
        
        Args:
            raw_features: Raw features
            roi_features: ROI features
            fusion_weights: Fusion weights
            
        Returns:
            Dictionary with contribution scores
        """
        # Compute feature magnitudes
        raw_magnitude = torch.norm(raw_features).item()
        roi_magnitude = torch.norm(roi_features).item()
        
        # Compute weight contributions
        # Fusion weights shape: [1, feature_dim, K+1]
        # First channel is for raw features, rest for ROIs
        weights = fusion_weights.squeeze(0)  # [feature_dim, K+1]
        raw_weight = weights[:, 0].mean().item()
        roi_weight = weights[:, 1:].mean().item()
        
        # Normalize contributions
        total_contribution = raw_magnitude * raw_weight + roi_magnitude * roi_weight
        if total_contribution > 0:
            raw_contribution = (raw_magnitude * raw_weight) / total_contribution
            roi_contribution = (roi_magnitude * roi_weight) / total_contribution
        else:
            raw_contribution = 0.5
            roi_contribution = 0.5
        
        return {
            'raw_contribution': float(raw_contribution),
            'roi_contribution': float(roi_contribution),
            'raw_magnitude': float(raw_magnitude),
            'roi_magnitude': float(roi_magnitude),
            'raw_weight': float(raw_weight),
            'roi_weight': float(roi_weight)
        }

