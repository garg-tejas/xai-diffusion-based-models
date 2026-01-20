"""
Score-CAM Explainer Module

Implements Score-CAM (Score-weighted Class Activation Mapping) for the DiffMIC-v2 model.
This is a gradient-free alternative to Grad-CAM that weights activation maps by their
importance scores obtained through forward passes.

Purpose:
- Generate class activation maps without requiring gradients
- More stable than gradient-based methods in some cases
- Reduce gradient saturation/vanishing issues

Method:
1. Extract activation maps from target layer (SamEncoder's final conv)
2. For each activation channel:
   a. Normalize activation map to [0, 1]
   b. Upsample to input resolution
   c. Use as soft mask on input image
   d. Forward pass masked image → get class score
3. Weight activation maps by score increase (importance)
4. Combine weighted activations into final saliency map

Reference:
- Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks
  (Wang et al., CVPR 2020 Workshops)
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from xai.core.base_explainer import BaseExplainer


class ScoreCAMExplainer(BaseExplainer):
    """
    Score-CAM explainer for DiffMIC-v2 model.
    
    This explainer generates class activation maps by weighting activation maps
    based on their contribution to the class score (via forward passes).
    
    Target Layer:
        - SamEncoder's final conv layer (model.encoder_x.g)
        - Extracts spatial activation maps before global pooling
        
    Attributes:
        model: CoolSystem model
        target_layer: Layer to extract activations from
        num_top_activations: Number of top activation channels to use
        activations: Cached activations from forward pass
        
    Usage:
        >>> explainer = ScoreCAMExplainer(model, device, config)
        >>> result = explainer.explain(image, label)
        >>> saliency = result['saliency_map']  # (H, W) numpy array
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device, config: Dict[str, Any]):
        """
        Initialize Score-CAM explainer.
        
        Args:
            model: The CoolSystem model
            device: Device to run on
            config: Configuration with Score-CAM settings
        """
        super().__init__(model, device, config)
        
        # Get configuration
        scorecam_config = config.get('explainers', {}).get('scorecam', {})
        target_layer_path = scorecam_config.get('target_layer', 'encoder_x.g')
        self.num_top_activations = scorecam_config.get('num_top_activations', None)  # None = use all
        
        # Get target layer
        self.target_layer = self._get_layer_by_path(target_layer_path)
        
        # Hook for capturing activations
        self.activations = None
        self._register_hook()
        
        print(f"[ScoreCAMExplainer] Initialized")
        print(f"  Device: {self.device}")
        print(f"  Target layer: {target_layer_path}")
        if self.num_top_activations:
            print(f"  Using top {self.num_top_activations} activation channels")
        else:
            print(f"  Using all activation channels")
    
    def _get_layer_by_path(self, path: str) -> nn.Module:
        """
        Get layer from model by dot-separated path.
        
        Args:
            path: Dot-separated path (e.g., 'encoder_x.g')
            
        Returns:
            Target layer module
        """
        # Use same logic as Grad-CAM - target auxiliary model's encoder
        parts = path.split('.')
        
        if path == 'encoder_x.g':
            # Target auxiliary model's encoder last conv layer
            try:
                layer = self.model.aux_model.encoder_global.f
                if hasattr(layer, '__getitem__'):
                    for i in range(len(layer) - 1, -1, -1):
                        if isinstance(layer[i], (nn.Conv2d, nn.BatchNorm2d)):
                            continue
                        if isinstance(layer[i], nn.Sequential):
                            for j in range(len(layer[i]) - 1, -1, -1):
                                if isinstance(layer[i][j], nn.Conv2d):
                                    return layer[i][j]
                    return layer[-1]
                return layer
            except AttributeError:
                for name, module in self.model.aux_model.named_modules():
                    if isinstance(module, nn.Conv2d):
                        last_conv = module
                return last_conv
        else:
            layer = self.model.aux_model
            for part in parts:
                layer = getattr(layer, part)
            return layer
    
    def _register_hook(self):
        """Register forward hook on target layer."""
        
        def forward_hook(module, input, output):
            """Save activations from forward pass."""
            self.activations = output
        
        self.target_layer.register_forward_hook(forward_hook)
    
    def _get_base_score(self, image: torch.Tensor, target_class: int) -> float:
        """
        Get base score for the target class without masking.
        
        Args:
            image: Input image (1, C, H, W)
            target_class: Target class index
            
        Returns:
            Base class score
        """
        with torch.no_grad():
            y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.model.aux_model(image)
            score = F.softmax(y0_aux_global, dim=1)[0, target_class].item()
        return score
    
    def _get_masked_score(self, image: torch.Tensor, mask: torch.Tensor, target_class: int) -> float:
        """
        Get class score for masked image.
        
        Args:
            image: Input image (1, C, H, W)
            mask: Soft mask (1, 1, H, W) in [0, 1]
            target_class: Target class index
            
        Returns:
            Class score for masked image
        """
        # Apply mask
        masked_image = image * mask
        
        with torch.no_grad():
            y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.model.aux_model(masked_image)
            score = F.softmax(y0_aux_global, dim=1)[0, target_class].item()
        
        return score
    
    def explain(self,
                image: torch.Tensor,
                label: Optional[int] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Generate Score-CAM explanation for a single image.
        
        Args:
            image: Input image tensor (C, H, W) or (1, C, H, W)
            label: Ground truth label (optional)
            **kwargs: Additional arguments
                - target_class: Class to generate CAM for (default: predicted class)
        
        Returns:
            Dictionary containing:
            - 'saliency_map': Score-CAM heatmap (H, W) in [0, 1]
            - 'prediction': Predicted class
            - 'confidence': Prediction confidence
            - 'target_class': Class used for CAM generation
            - 'importance_scores': Score for each activation channel
        """
        # Preprocess
        image = self._preprocess_image(image)
        target_class = kwargs.get('target_class', None)
        
        # Reset activations
        self.activations = None
        
        # Get prediction and activations
        with torch.no_grad():
            y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.model.aux_model(image)
            probs = F.softmax(y0_aux_global, dim=1)
            prediction = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0, prediction].item())
        
        # Check if activations were captured
        if self.activations is None:
            raise RuntimeError("Activations not captured! Hook may not be properly registered.")
        
        # Determine target class
        if target_class is None:
            target_class = prediction
        
        # Get activations (should be cached from forward pass)
        activations = self.activations  # (1, C, H, W)
        batch_size, num_channels, height, width = activations.shape
        
        # Select top-k activation channels if specified
        if self.num_top_activations and self.num_top_activations < num_channels:
            # Select channels with highest activation magnitudes
            activation_magnitudes = activations.abs().mean(dim=[0, 2, 3])  # (C,)
            top_indices = torch.topk(activation_magnitudes, self.num_top_activations).indices
            activations = activations[:, top_indices, :, :]
            num_channels = self.num_top_activations
        
        # Get base score (no masking)
        base_score = self._get_base_score(image, target_class)
        
        # Compute importance score for each activation channel
        importance_scores = []
        
        for i in range(num_channels):
            # Extract single activation map
            activation_map = activations[0, i:i+1, :, :]  # (1, 1, H, W)
            
            # Normalize to [0, 1]
            if activation_map.max() > activation_map.min():
                activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
            else:
                activation_map = torch.zeros_like(activation_map)
            
            # Upsample to input resolution
            mask = F.interpolate(activation_map.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False)
            mask = mask.squeeze(0)  # (1, 1, H, W)
            
            # Get score with this mask
            masked_score = self._get_masked_score(image, mask, target_class)
            
            # Compute importance as score increase (can be negative)
            importance = max(masked_score - base_score, 0)  # Only positive contributions
            importance_scores.append(importance)
        
        # Convert importance scores to weights
        importance_scores = np.array(importance_scores)
        if importance_scores.sum() > 0:
            weights = importance_scores / importance_scores.sum()
        else:
            weights = np.ones(num_channels) / num_channels
        
        # Compute weighted combination of activation maps
        cam = torch.zeros((height, width), dtype=torch.float32).to(self.device)
        
        for i, weight in enumerate(weights):
            activation_map = activations[0, i, :, :]
            cam += weight * activation_map
        
        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Upsample to input resolution
        cam = cam.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        cam = F.interpolate(cam, size=(512, 512), mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        
        # Build explanation
        explanation = self.get_explanation_dict(
            explanation_type='scorecam',
            prediction=prediction,
            confidence=confidence,
            saliency_map=cam,
            target_class=target_class,
            importance_scores=importance_scores.tolist(),
            num_channels_used=num_channels,
            ground_truth=label if label is not None else -1,
        )
        
        # Cleanup
        self.activations = None
        torch.cuda.empty_cache()
        
        return explanation
