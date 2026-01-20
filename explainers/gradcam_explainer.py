"""
Grad-CAM Explainer Module

Implements Gradient-weighted Class Activation Mapping (Grad-CAM) for the DiffMIC-v2 model.
This method visualizes which regions of the input image are important for classification
by computing gradients of the predicted class score with respect to feature maps.

Purpose:
- Generate class-discriminative localization maps
- Highlight important regions without training additional models
- Provide interpretable saliency maps for medical image diagnosis

Method:
1. Forward pass: Extract feature maps from target layer (SamEncoder's final conv)
2. Backward pass: Compute gradients of class score w.r.t. feature maps
3. Weight feature maps by gradient importance (global average pooling of gradients)
4. Generate heatmap via weighted sum + ReLU
5. Upsample to input resolution

Reference:
- Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
  (Selvaraju et al., ICCV 2017)
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from xai.core.base_explainer import BaseExplainer


class GradCAMExplainer(BaseExplainer):
    """
    Grad-CAM explainer for DiffMIC-v2 model.
    
    This explainer generates class activation maps by computing gradients
    of the predicted class score with respect to convolutional feature maps
    in the SamEncoder pathway.
    
    Target Layer:
        - SamEncoder's final conv layer (model.encoder_x.g)
        - Shape: (batch, feature_dim, H, W) where H=W=32 for 512x512 input
        
    Attributes:
        model: CoolSystem model
        target_layer: Layer to extract gradients from
        feature_maps: Cached feature maps from forward pass
        gradients: Cached gradients from backward pass
        
    Usage:
        >>> explainer = GradCAMExplainer(model, device, config)
        >>> result = explainer.explain(image, label)
        >>> saliency = result['saliency_map']  # (H, W) numpy array
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device, config: Dict[str, Any]):
        """
        Initialize Grad-CAM explainer.
        
        Args:
            model: The CoolSystem model
            device: Device to run on
            config: Configuration with Grad-CAM settings
        """
        super().__init__(model, device, config)
        
        # Extract target layer path from config
        target_layer_path = config.get('explainers', {}).get('gradcam', {}).get('target_layer', 'encoder_x.g')
        
        # Get target layer from model
        self.target_layer = self._get_layer_by_path(target_layer_path)
        
        # Hooks for capturing activations and gradients
        self.feature_maps = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
        
        print(f"[GradCAMExplainer] Initialized")
        print(f"  Device: {self.device}")
        print(f"  Target layer: {target_layer_path}")
        print(f"  Target layer shape: {self._get_layer_output_shape()}")
    
    def _get_layer_by_path(self, path: str) -> nn.Module:
        """
        Get layer from model by dot-separated path.
        
        Args:
            path: Dot-separated path (e.g., 'encoder_x.g')
            
        Returns:
            Target layer module
        """
        # For DiffMIC-v2, we need to target the auxiliary model's encoder
        # since that's what processes the image in our forward pass
        # The auxiliary model (DCG) has its own ResNet encoder
        
        # Path format: "layer_name" or "module.layer_name"
        parts = path.split('.')
        
        # Start from auxiliary model's DCG structure
        # The DCG model has encoder_global (ResNet) that we can target
        if path == 'encoder_x.g':
            # This is trying to access ConditionalModel's encoder
            # But we use aux_model, so redirect to aux_model's encoder
            # Aux model structure: model.aux_model (DCG) -> encoder_global (ResNet)
            try:
                # Try auxiliary model's global encoder last conv layer
                layer = self.model.aux_model.encoder_global.f
                # Get the last conv layer from ResNet
                if hasattr(layer, '__getitem__'):
                    # Sequential - get last layer before pooling
                    for i in range(len(layer) - 1, -1, -1):
                        if isinstance(layer[i], (nn.Conv2d, nn.BatchNorm2d)):
                            continue
                        if isinstance(layer[i], nn.Sequential):
                            # Found a Sequential block, get its last conv
                            for j in range(len(layer[i]) - 1, -1, -1):
                                if isinstance(layer[i][j], nn.Conv2d):
                                    return layer[i][j]
                    # If not found in Sequential blocks, use last layer
                    return layer[-1]
                return layer
            except AttributeError:
                # Fallback: try to find any conv layer in aux model
                for name, module in self.model.aux_model.named_modules():
                    if isinstance(module, nn.Conv2d):
                        last_conv = module  # Keep updating to get the last one
                return last_conv
        else:
            # Generic path traversal
            layer = self.model.aux_model
            for part in parts:
                layer = getattr(layer, part)
            return layer
    
    def _get_layer_output_shape(self) -> str:
        """Get string representation of target layer output shape."""
        try:
            # Create dummy input
            dummy_input = torch.randn(1, 3, 512, 512).to(self.device)
            with torch.no_grad():
                _ = self.model.model.encoder_x(dummy_input)
            if self.feature_maps is not None:
                return str(self.feature_maps.shape)
            return "Unknown"
        except:
            return "Unknown"
    
    def _register_hooks(self):
        """Register forward hook on target layer."""
        
        def forward_hook(module, input, output):
            """Save feature maps and enable gradient tracking."""
            self.feature_maps = output
            if output.requires_grad:
                output.retain_grad()
        
        # Only register forward hook - we'll access gradients via .grad attribute
        self.target_layer.register_forward_hook(forward_hook)
    
    def explain(self,
                image: torch.Tensor,
                label: Optional[int] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Generate Grad-CAM explanation for a single image.
        
        Args:
            image: Input image tensor (C, H, W) or (1, C, H, W)
            label: Ground truth label (optional)
            **kwargs: Additional arguments
                - target_class: Class to generate CAM for (default: predicted class)
        
        Returns:
            Dictionary containing:
            - 'saliency_map': Grad-CAM heatmap (H, W) in [0, 1]
            - 'prediction': Predicted class
            - 'confidence': Prediction confidence
            - 'target_class': Class used for CAM generation
            - 'raw_cam': Pre-upsampled CAM (feature map resolution)
        """
        # Preprocess
        image = self._preprocess_image(image)
        target_class = kwargs.get('target_class', None)
        
        # Create a leaf tensor with gradients enabled
        image_leaf = image.clone().detach().requires_grad_(True)
        
        # Keep model in eval mode for consistent predictions
        # But ensure gradients can flow
        original_mode = self.model.training
        self.model.eval()  # Use eval mode for consistent predictions
        
        # Reset hooks
        self.feature_maps = None
        self.gradients = None
        
        try:
            # Forward pass through auxiliary model (with hooks to capture feature maps)
            y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.model.aux_model(image_leaf)
            
            # Use auxiliary model's global prediction
            probs = F.softmax(y0_aux_global, dim=1)
            prediction = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0, prediction].item())
            
            # Determine target class for CAM
            if target_class is None:
                target_class = prediction
            
            # Check if feature maps were captured
            if self.feature_maps is None:
                raise RuntimeError("Feature maps not captured! Hook may not be properly registered.")
            
            # Ensure feature maps require gradients
            if not self.feature_maps.requires_grad:
                raise RuntimeError("Feature maps don't require gradients! Check hook implementation.")
            
            # Get class score and compute gradients via backward pass
            class_score = y0_aux_global[0, target_class]
            
            # Zero gradients first
            self.model.zero_grad()
            if image_leaf.grad is not None:
                image_leaf.grad.zero_()
            
            # Backward pass to compute gradients
            class_score.backward()

            # Get gradients from feature maps (captured via retain_grad())
            if self.feature_maps.grad is None:
                raise RuntimeError("Gradients not computed on feature maps! Backward pass may have failed.")
            
            gradients = self.feature_maps.grad.detach()  # (1, C, H, W)
            feature_maps = self.feature_maps.detach()  # (1, C, H, W)
            
            # Proper Grad-CAM weights using Global Average Pooling of gradients
            # This is the key difference from CAM - we weight by gradient importance
            weights = gradients.mean(dim=[2, 3])[0]  # (C,) - average over spatial dimensions
            
            # Weighted combination of feature maps
            # cam shape: (H, W)
            cam = torch.zeros(feature_maps.shape[2:], dtype=torch.float32).to(self.device)
            for i, w in enumerate(weights):
                cam += w * feature_maps[0, i, :, :]
            
            # Apply ReLU (only positive contributions)
            cam = F.relu(cam)
            
            # Normalize to [0, 1]
            if cam.max() > 0:
                cam = cam / cam.max()
            
            # Save raw CAM before upsampling
            raw_cam = cam.detach().cpu().numpy()
            
            # Upsample to input resolution
            cam = cam.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            cam = F.interpolate(cam, size=(512, 512), mode='bilinear', align_corners=False)
            cam = cam.squeeze().detach().cpu().numpy()
            
        finally:
            # Restore original mode
            self.model.train(original_mode)
        
        # Build explanation
        explanation = self.get_explanation_dict(
            explanation_type='gradcam',
            prediction=prediction,
            confidence=confidence,
            saliency_map=cam,
            target_class=target_class,
            raw_cam=raw_cam,
            ground_truth=label if label is not None else -1,
        )
        
        # Cleanup
        self.feature_maps = None
        self.gradients = None
        torch.cuda.empty_cache()
        
        return explanation
    
    def generate_overlay(self, image: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Generate overlay of CAM on original image.
        
        Args:
            image: Original image (H, W, 3) in [0, 255]
            cam: CAM heatmap (H, W) in [0, 1]
            alpha: Overlay transparency
            
        Returns:
            Overlay image (H, W, 3) in [0, 255]
        """
        # Convert CAM to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Ensure image is uint8
        if image.max() <= 1.0:
            image = np.uint8(255 * image)
        
        # Overlay
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return overlay
