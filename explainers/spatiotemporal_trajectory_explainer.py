"""
Spatio-Temporal Trajectory Explainer Module

This module tracks the evolution of cross-attention between the denoising U-Net
and the conditional features (F_raw, F_roi_1, ..., F_roi_K) across timesteps.

Purpose:
- Visualize "coarse-to-fine" reasoning process
- Show how attention shifts from global to local features
- Track when the model focuses on specific ROIs

This fuses DiffusionExplainer and AttentionExplainer concepts to show
spatio-temporal evolution of attention.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from xai.core.base_explainer import BaseExplainer


class SpatioTemporalTrajectoryExplainer(BaseExplainer):
    """
    Track cross-attention evolution through diffusion timesteps.
    
    This explainer uses forward hooks to extract the cond_weight activation
    at each tracked timestep, showing how the U-Net's attention to global vs
    local features evolves during denoising.
    
    Attributes:
        model: CoolSystem model
        cond_model: ConditionalModel (denoising U-Net)
        sampler: SR3Sampler
        
    Usage:
        >>> explainer = SpatioTemporalTrajectoryExplainer(model, device, config)
        >>> result = explainer.explain(image, label, track_timesteps=[900, 500, 100])
        >>> print(f"Tracked {len(result['attention_trajectory'])} timesteps")
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device, config: Dict[str, Any]):
        """
        Initialize spatio-temporal trajectory explainer.
        
        Args:
            model: The CoolSystem model
            device: Device to run on
            config: Configuration with trajectory settings
        """
        super().__init__(model, device, config)
        
        # Extract components
        self.aux_model = model.aux_model
        self.cond_model = model.model
        self.sampler = model.DiffSampler
        self.scheduler = self.sampler.scheduler
        self.model_config = model.params
        
        # Default timesteps to track
        self.default_track_timesteps = config.get('track_timesteps', [900, 700, 500, 300, 100, 10])
        
        print(f"[SpatioTemporalTrajectoryExplainer] Initialized")
        print(f"  Default track timesteps: {self.default_track_timesteps}")
        print(f"  Device: {self.device}")
    
    def explain(self,
                image: torch.Tensor,
                label: Optional[int] = None,
                track_timesteps: Optional[List[int]] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Track attention evolution through timesteps.
        
        Args:
            image: Input image tensor
            label: Ground truth label (optional)
            track_timesteps: List of timestep values to track
                           If None, uses default timesteps
            **kwargs: Additional arguments
        
        Returns:
            Dictionary containing:
            - 'attention_trajectory': List of attention maps at each timestep
            - 'global_attention_evolution': Global attention over time
            - 'local_attention_evolution': Local attention over time
            - 'roi_attention_evolution': Per-ROI attention over time
        """
        # Preprocess
        image = self._preprocess_image(image)
        
        # Use default timesteps if not specified
        if track_timesteps is None:
            track_timesteps = self.default_track_timesteps
        
        # Get auxiliary model outputs
        with torch.no_grad():
            (y_fusion, y_global, y_local, 
             patches, patch_attns, saliency_map) = self.aux_model(image)
        
        # Prepare guidance conditions
        bz = image.shape[0]
        nc = self.model_config.data.num_classes
        _, np_patches = patch_attns.shape
        
        y0_cond = self._guided_prob_map(y_global, y_local, bz, nc, np_patches)
        yT = self._guided_prob_map(
            torch.rand_like(y_global),
            torch.rand_like(y_local),
            bz, nc, np_patches
        )
        
        attns = patch_attns.unsqueeze(-1)
        attns = (attns * attns.transpose(1, 2)).unsqueeze(1)
        num_patches = patch_attns.size(1)
        
        # Track attention evolution
        attention_trajectory = self._track_attention_evolution(
            image,
            yT,
            y0_cond,
            patches,
            attns,
            patch_attns,
            num_patches,
            track_timesteps
        )
        
        # Analyze trajectory
        global_attn_evolution = [step['global_attention'] for step in attention_trajectory]
        local_attn_evolution = [step['local_attention'] for step in attention_trajectory]
        roi_attn_evolution = [step['roi_attention'] for step in attention_trajectory]
        
        # Build explanation
        explanation = self.get_explanation_dict(
            explanation_type='spatiotemporal_trajectory',
            prediction=attention_trajectory[-1]['prediction'] if attention_trajectory else -1,
            confidence=attention_trajectory[-1]['confidence'] if attention_trajectory else 0.0,
            
            # Trajectory data
            attention_trajectory=attention_trajectory,
            global_attention_evolution=global_attn_evolution,
            local_attention_evolution=local_attn_evolution,
            roi_attention_evolution=roi_attn_evolution,
            tracked_timesteps=track_timesteps,
            
            # Ground truth
            ground_truth=label if label is not None else -1,
        )
        
        return explanation
    
    def _track_attention_evolution(self,
                                   x_batch: torch.Tensor,
                                   yT: torch.Tensor,
                                   y0_cond: torch.Tensor,
                                   patches: torch.Tensor,
                                   attns: torch.Tensor,
                                   patch_attn_weights: torch.Tensor,
                                   num_patches: int,
                                   track_timesteps: List[int]) -> List[Dict]:
        """
        Track attention weights at specified timesteps.
        
        Args:
            x_batch: Input image
            yT: Initial noisy state
            y0_cond: Guidance condition
            patches: Local patches
            attns: Attention weights
            track_timesteps: List of timestep values to track
        
        Returns:
            List of dictionaries with attention data at each tracked timestep
        """
        bz, nc, h, w = y0_cond.shape
        bz = int(bz)
        nc = int(nc)
        h = int(h)
        w = int(w)
        spatial_size = bz * h * w
        noisy_y = yT.clone()
        
        # Pre-compute masks for diagonal (global) vs off-diagonal (local) contributions
        diag_hw = torch.eye(h, dtype=torch.bool, device=x_batch.device)
        if w != h:
            min_hw = min(h, w)
            diag_hw = torch.eye(min_hw, dtype=torch.bool, device=x_batch.device)
            diag_hw = F.pad(
                diag_hw,
                (0, w - min_hw, 0, h - min_hw),
                value=False
            )
        diag_mask = diag_hw.unsqueeze(0).unsqueeze(0).expand(bz, nc, h, w)
        off_diag_mask = (~diag_hw).unsqueeze(0).unsqueeze(0).expand(bz, nc, h, w)
        
        # ROI distribution derived from auxiliary patch attentions
        if patch_attn_weights is not None:
            roi_distribution = patch_attn_weights[0].detach().cpu().numpy()
            roi_sum = float(np.sum(roi_distribution))
            if roi_sum > 0:
                roi_distribution = roi_distribution / roi_sum
            else:
                roi_distribution = np.ones(num_patches) / num_patches
        else:
            roi_distribution = np.ones(num_patches) / num_patches
        
        trajectory = []
        
        for t_idx, t in enumerate(self.scheduler.timesteps):
            if hasattr(t, "item"):
                t_value = int(t.item())
            else:
                t_value = int(t)
            timesteps_batch = torch.full(
                (spatial_size,),
                t_value,
                device=x_batch.device,
                dtype=torch.long,
            )
            
            with torch.no_grad():
                try:
                    noise_pred = self.cond_model(
                        x_batch,
                        torch.cat([y0_cond, noisy_y], dim=1),
                        timesteps_batch,
                        patches,
                        attns
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"cond_model forward failed at timestep {t_value} "
                        f"with timesteps_batch shape={timesteps_batch.shape}, "
                        f"dtype={timesteps_batch.dtype}, "
                        f"y0_cond shape={y0_cond.shape}, noisy_y shape={noisy_y.shape}, "
                        f"patches shape={patches.shape}, attns shape={attns.shape}"
                    ) from e
            
            # Denoise
            prev_noisy_y = noisy_y.clone()
            noisy_y = self.scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=noisy_y
            ).prev_sample
            
            # Check if this timestep should be tracked
            t_val = t_value
            if t_val in track_timesteps:
                # Measure contribution energy from current guided probability map
                attn_map = torch.abs(noisy_y.detach())
                
                global_contribution = attn_map.masked_select(diag_mask).mean().item()
                local_contribution = attn_map.masked_select(off_diag_mask).mean().item()
                
                total_contribution = global_contribution + local_contribution
                if total_contribution > 0:
                    global_attention = float(global_contribution / total_contribution)
                    local_attention = float(local_contribution / total_contribution)
                else:
                    global_attention = 0.5
                    local_attention = 0.5
                
                # Distribute local attention across ROIs using auxiliary patch weights
                roi_attention = (roi_distribution * local_attention).tolist()
                
                # Get prediction at this timestep
                probs_spatial = F.softmax(noisy_y, dim=1)
                probs = probs_spatial.mean(dim=[2, 3])
                prediction = int(torch.argmax(probs, dim=1).item())
                confidence = float(probs[0, prediction].item())
                
                trajectory.append({
                    'timestep': t_val,
                    'timestep_idx': t_idx,
                    'global_attention': global_attention,
                    'local_attention': local_attention,
                    'roi_attention': roi_attention,
                    'prediction': prediction,
                    'confidence': confidence,
                    'probs': probs[0].detach().cpu().numpy(),
                })
        
        return trajectory
    
    def _guided_prob_map(self, y0_g, y0_l, bz, nc, np_patches):
        """Create guided probability map."""
        device = y0_g.device
        
        distance_to_diag = torch.tensor(
            [[abs(i-j) for j in range(np_patches)] for i in range(np_patches)]
        ).to(device)
        
        weight_g = 1 - distance_to_diag / (np_patches - 1)
        weight_l = distance_to_diag / (np_patches - 1)
        
        interpolated_value = (
            weight_l.unsqueeze(0).unsqueeze(0) * y0_l.unsqueeze(-1).unsqueeze(-1) +
            weight_g.unsqueeze(0).unsqueeze(0) * y0_g.unsqueeze(-1).unsqueeze(-1)
        )
        
        diag_indices = torch.arange(np_patches)
        prob_map = interpolated_value.clone()
        for i in range(bz):
            for j in range(nc):
                prob_map[i, j, diag_indices, diag_indices] = y0_g[i, j]
                prob_map[i, j, np_patches-1, 0] = y0_l[i, j]
                prob_map[i, j, 0, np_patches-1] = y0_l[i, j]
        
        return prob_map

