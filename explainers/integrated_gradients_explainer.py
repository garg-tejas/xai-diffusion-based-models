"""
Integrated Gradients Explainer Module

Implements Integrated Gradients (IG) for the DiffMIC-v2 model.
This method attributes the prediction to input features by integrating gradients
along a straight path from a baseline (black image) to the actual input.

Purpose:
- Provide pixel-level attribution scores
- Satisfy axioms: Sensitivity and Implementation Invariance
- Generate fine-grained explanations for medical images

Method:
1. Define baseline: x' (typically black image or blurred version)
2. Create interpolated images: x_i = x' + (i/m) * (x - x') for i=1..m
3. Compute gradients: ∇f(x_i) for each interpolated image
4. Integrate: IG(x) = (x - x') ⊙ Σ(∇f(x_i) / m)
5. Return pixel-wise attribution map

Reference:
- Axiomatic Attribution for Deep Networks (Sundararajan et al., ICML 2017)
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


class IntegratedGradientsExplainer(BaseExplainer):
    """
    Integrated Gradients explainer for DiffMIC-v2 model.
    
    This explainer computes pixel-level importance by integrating gradients
    along the path from a baseline to the input image.
    
    Baseline Options:
        - 'black': Zero tensor (default for medical images)
        - 'blur': Gaussian blurred version of input
        - 'mean': Mean pixel value
        
    Attributes:
        model: CoolSystem model
        num_steps: Number of interpolation steps (default: 50)
        baseline_type: Type of baseline to use
        
    Usage:
        >>> explainer = IntegratedGradientsExplainer(model, device, config)
        >>> result = explainer.explain(image, label)
        >>> attribution = result['attribution_map']  # (H, W, C) numpy array
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device, config: Dict[str, Any]):
        """
        Initialize Integrated Gradients explainer.
        
        Args:
            model: The CoolSystem model
            device: Device to run on
            config: Configuration with IG settings
        """
        super().__init__(model, device, config)
        
        # Get configuration
        ig_config = config.get('explainers', {}).get('integrated_gradients', {})
        self.num_steps = ig_config.get('num_steps', 50)
        self.baseline_type = ig_config.get('baseline', 'black')
        
        print(f"[IntegratedGradientsExplainer] Initialized")
        print(f"  Device: {self.device}")
        print(f"  Num steps: {self.num_steps}")
        print(f"  Baseline: {self.baseline_type}")
    
    def _create_baseline(self, image: torch.Tensor) -> torch.Tensor:
        """
        Create baseline image based on configuration.
        
        Args:
            image: Input image (1, C, H, W)
            
        Returns:
            Baseline image of same shape
        """
        if self.baseline_type == 'black':
            # Zero tensor (black image)
            baseline = torch.zeros_like(image)
        
        elif self.baseline_type == 'blur':
            # Gaussian blur
            import torchvision.transforms.functional as TF
            from PIL import Image
            
            # Convert to PIL, blur, convert back
            img_np = image[0].cpu().permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            img_blurred = TF.gaussian_blur(TF.to_tensor(img_pil).unsqueeze(0), kernel_size=51, sigma=20)
            baseline = img_blurred.to(self.device)
        
        elif self.baseline_type == 'mean':
            # Mean pixel value
            mean_val = image.mean()
            baseline = torch.full_like(image, mean_val)
        
        else:
            # Default to black
            baseline = torch.zeros_like(image)
        
        return baseline
    
    def _interpolate_images(self, baseline: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Create interpolated images between baseline and input.
        
        Args:
            baseline: Baseline image (1, C, H, W)
            image: Input image (1, C, H, W)
            
        Returns:
            Interpolated images (num_steps, C, H, W)
        """
        # Create alpha values for interpolation
        alphas = torch.linspace(0, 1, self.num_steps + 1, device=self.device)
        
        # Interpolate: x_i = baseline + alpha_i * (image - baseline)
        interpolated = []
        for alpha in alphas:
            interpolated_img = baseline + alpha * (image - baseline)
            interpolated.append(interpolated_img)
        
        # Stack into batch
        interpolated = torch.cat(interpolated, dim=0)  # (num_steps+1, C, H, W)
        
        return interpolated, alphas
    
    def _compute_gradients_batch(self, images: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        Compute gradients for a batch of interpolated images.
        
        Args:
            images: Batch of interpolated images (B, C, H, W)
            target_class: Class to compute gradients for
            
        Returns:
            Gradients w.r.t. input (B, C, H, W)
        """
        batch_size = images.shape[0]
        
        # Clone and detach to create leaf variable
        images_leaf = images.clone().detach().requires_grad_(True)
        
        # Forward pass through auxiliary model (using as proxy for efficiency)
        y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.model.aux_model(images_leaf)
        
        # Get class scores
        class_scores = y0_aux_global[:, target_class]
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=class_scores,
            inputs=images_leaf,
            grad_outputs=torch.ones_like(class_scores),
            create_graph=False,
            retain_graph=False
        )[0]
        
        return gradients
    
    def explain(self,
                image: torch.Tensor,
                label: Optional[int] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Generate Integrated Gradients explanation for a single image.
        
        Args:
            image: Input image tensor (C, H, W) or (1, C, H, W)
            label: Ground truth label (optional)
            **kwargs: Additional arguments
                - target_class: Class to generate attribution for (default: predicted class)
                - batch_size: Batch size for gradient computation (default: 10)
        
        Returns:
            Dictionary containing:
            - 'attribution_map': Pixel-wise attribution (H, W, C) in [-1, 1]
            - 'saliency_map': Absolute attribution summed over channels (H, W)
            - 'prediction': Predicted class
            - 'confidence': Prediction confidence
            - 'target_class': Class used for attribution
        """
        # Preprocess
        image = self._preprocess_image(image)
        target_class = kwargs.get('target_class', None)
        batch_size = kwargs.get('batch_size', 10)
        
        # Get prediction first
        with torch.no_grad():
            y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.model.aux_model(image)
            probs = F.softmax(y0_aux_global, dim=1)
            prediction = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0, prediction].item())
        
        # Determine target class
        if target_class is None:
            target_class = prediction
        
        # Create baseline
        baseline = self._create_baseline(image)
        
        # Create interpolated images
        interpolated, alphas = self._interpolate_images(baseline, image)
        
        # Compute gradients for all interpolated images (with batching)
        all_gradients = []
        num_batches = (len(interpolated) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(interpolated))
            batch = interpolated[start_idx:end_idx]
            
            with torch.enable_grad():
                grads = self._compute_gradients_batch(batch, target_class)
                all_gradients.append(grads.detach())
            
            # Clear cache
            torch.cuda.empty_cache()
        
        # Concatenate all gradients
        all_gradients = torch.cat(all_gradients, dim=0)  # (num_steps+1, C, H, W)
        
        # Compute integrated gradients using trapezoidal rule
        # IG = (x - baseline) * average_gradient
        avg_gradients = all_gradients.mean(dim=0, keepdim=True)  # (1, C, H, W)
        integrated_gradients = (image - baseline) * avg_gradients
        
        # Convert to numpy
        attribution_map = integrated_gradients[0].detach().cpu().numpy()  # (C, H, W)
        attribution_map = attribution_map.transpose(1, 2, 0)  # (H, W, C)
        
        # Create saliency map (sum absolute attributions over channels)
        saliency_map = np.abs(attribution_map).sum(axis=2)  # (H, W)
        
        # Normalize saliency to [0, 1]
        if saliency_map.max() > 0:
            saliency_map = saliency_map / saliency_map.max()
        
        # Build explanation
        explanation = self.get_explanation_dict(
            explanation_type='integrated_gradients',
            prediction=prediction,
            confidence=confidence,
            attribution_map=attribution_map,
            saliency_map=saliency_map,
            target_class=target_class,
            num_steps=self.num_steps,
            baseline_type=self.baseline_type,
            ground_truth=label if label is not None else -1,
        )
        
        # Cleanup
        torch.cuda.empty_cache()
        
        return explanation
    
    def get_positive_attribution(self, attribution_map: np.ndarray) -> np.ndarray:
        """
        Extract positive attributions (features that support the prediction).
        
        Args:
            attribution_map: Full attribution map (H, W, C)
            
        Returns:
            Positive attribution map (H, W)
        """
        positive = np.maximum(attribution_map, 0)
        return positive.sum(axis=2)
    
    def get_negative_attribution(self, attribution_map: np.ndarray) -> np.ndarray:
        """
        Extract negative attributions (features that oppose the prediction).
        
        Args:
            attribution_map: Full attribution map (H, W, C)
            
        Returns:
            Negative attribution map (H, W)
        """
        negative = np.minimum(attribution_map, 0)
        return np.abs(negative).sum(axis=2)
