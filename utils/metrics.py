"""
XAI Metrics Module

Implements quantitative evaluation metrics for assessing the faithfulness and quality
of saliency maps and attribution methods.

Metrics:
1. Insertion AUC: Measures how quickly confidence increases when revealing important pixels
2. Deletion AUC: Measures how quickly confidence decreases when removing important pixels  
3. Stability: Measures robustness of explanations to small input perturbations

Purpose:
- Quantitatively evaluate XAI method quality
- Compare different explanation techniques
- Validate that saliency maps actually identify important regions

References:
- RISE: Randomized Input Sampling for Explanation (Petsiuk et al., BMVC 2018)
- Sanity Checks for Saliency Maps (Adebayo et al., NeurIPS 2018)
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable
from scipy.ndimage import gaussian_filter
from sklearn.metrics import auc
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def compute_insertion_auc(
    model: nn.Module,
    image: torch.Tensor,
    saliency_map: np.ndarray,
    target_class: int,
    steps: int = 20,
    occlusion_method: str = 'blur',
    device: torch.device = None,
    precision_mode: str = 'balanced'
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Insertion AUC metric.
    
    Progressively reveals pixels in order of importance (high saliency first)
    and measures how quickly the model's confidence increases.
    
    Args:
        model: The trained model (CoolSystem)
        image: Input image (1, C, H, W) tensor
        saliency_map: Saliency map (H, W) in [0, 1]
        target_class: Class to measure confidence for
        steps: Number of insertion steps
        occlusion_method: 'blur' or 'zero' for occluded pixels
        device: Device to run on
        
    Returns:
        - auc_score: Area under insertion curve (higher is better)
        - fractions: Array of pixel fractions revealed
        - confidences: Corresponding confidence scores
    """
    if device is None:
        device = image.device
    
    # Start with fully occluded image
    if occlusion_method == 'blur':
        occluded_image = _blur_image(image, sigma=10.0)
    else:
        occluded_image = torch.zeros_like(image)
    
    # Use precision mode to determine region size
    precision_config = {
        "fast": {"region_size": 8},      # 64x faster, lowest quality
        "balanced": {"region_size": 4},   # 16x faster, good quality
        "precise": {"region_size": 1},   # Pixel-level, best quality
    }
    
    config_params = precision_config.get(precision_mode, precision_config["balanced"])
    region_size = config_params["region_size"]
    
    H, W = saliency_map.shape
    small_H, small_W = max(1, H // region_size), max(1, W // region_size)
    
    # Downsample saliency map to small grid
    small_saliency = cv2.resize(saliency_map, (small_W, small_H), interpolation=cv2.INTER_AREA)
    
    # Get region coordinates sorted by saliency (descending)
    region_coords = _get_sorted_pixel_coords(small_saliency, descending=True)
    total_regions = len(region_coords)
    
    # Track confidence at each step
    fractions = []
    confidences = []
    
    # Iterate through insertion steps
    current_image = occluded_image.clone()
    
    for step in range(steps + 1):
        # Get confidence for current image
        confidence = _get_confidence(model, current_image, target_class, device)
        
        # Record results
        fraction = step / steps
        fractions.append(fraction)
        confidences.append(confidence)
        
        # Reveal next batch of regions (if not last step)
        if step < steps:
            num_regions_to_reveal = int((step + 1) * total_regions / steps)
            regions_to_reveal = region_coords[:num_regions_to_reveal]
            current_image = _reveal_regions(current_image, image, regions_to_reveal, region_size)
    
    # Compute AUC
    fractions = np.array(fractions)
    confidences = np.array(confidences)
    auc_score = auc(fractions, confidences)
    
    return auc_score, fractions, confidences


def compute_deletion_auc(
    model: nn.Module,
    image: torch.Tensor,
    saliency_map: np.ndarray,
    target_class: int,
    steps: int = 20,
    occlusion_method: str = 'blur',
    device: torch.device = None,
    precision_mode: str = 'balanced'
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Deletion AUC metric.
    
    Progressively removes pixels in order of importance (high saliency first)
    and measures how quickly the model's confidence decreases.
    
    Args:
        model: The trained model (CoolSystem)
        image: Input image (1, C, H, W) tensor
        saliency_map: Saliency map (H, W) in [0, 1]
        target_class: Class to measure confidence for
        steps: Number of deletion steps
        occlusion_method: 'blur' or 'zero' for deleted pixels
        device: Device to run on
        
    Returns:
        - auc_score: Area under deletion curve (lower is better)
        - fractions: Array of pixel fractions removed
        - confidences: Corresponding confidence scores
    """
    if device is None:
        device = image.device
    
    # Start with full image
    current_image = image.clone()
    
    # Use precision mode to determine region size
    precision_config = {
        "fast": {"region_size": 8},
        "balanced": {"region_size": 4},
        "precise": {"region_size": 1},
    }
    
    config_params = precision_config.get(precision_mode, precision_config["balanced"])
    region_size = config_params["region_size"]
    
    H, W = saliency_map.shape
    small_H, small_W = max(1, H // region_size), max(1, W // region_size)
    
    # Downsample saliency map to small grid
    small_saliency = cv2.resize(saliency_map, (small_W, small_H), interpolation=cv2.INTER_AREA)
    
    # Get region coordinates sorted by saliency (descending)
    region_coords = _get_sorted_pixel_coords(small_saliency, descending=True)
    total_regions = len(region_coords)
    
    # Track confidence at each step
    fractions = []
    confidences = []
    
    # Iterate through deletion steps
    for step in range(steps + 1):
        # Get confidence for current image
        confidence = _get_confidence(model, current_image, target_class, device)
        
        # Record results
        fraction = step / steps
        fractions.append(fraction)
        confidences.append(confidence)
        
        # Delete next batch of regions (if not last step)
        if step < steps:
            num_regions_to_delete = int((step + 1) * total_regions / steps)
            regions_to_delete = region_coords[:num_regions_to_delete]
            current_image = _occlude_regions(image, regions_to_delete, region_size, occlusion_method)
    
    # Compute AUC
    fractions = np.array(fractions)
    confidences = np.array(confidences)
    auc_score = auc(fractions, confidences)
    
    return auc_score, fractions, confidences


def compute_stability(
    explainer: Any,
    image: torch.Tensor,
    num_perturbations: int = 20,
    perturbation_magnitude: float = 0.05,
    device: torch.device = None
) -> Tuple[float, list]:
    """
    Compute Stability metric.
    
    Measures how consistent the explanation is when the input is slightly perturbed.
    Higher stability means the explanation is more robust.
    
    Args:
        explainer: XAI explainer instance (must have explain() method)
        image: Input image (1, C, H, W) tensor
        num_perturbations: Number of perturbed versions to generate
        perturbation_magnitude: Magnitude of perturbations (0-1)
        device: Device to run on
        
    Returns:
        - stability_score: Average similarity between original and perturbed explanations
        - similarities: List of similarity scores for each perturbation
    """
    if device is None:
        device = image.device
    
    # Get explanation for original image
    original_explanation = explainer.explain(image)
    original_saliency = original_explanation['saliency_map']
    
    # Generate perturbed versions and compute explanations
    similarities = []
    
    for i in range(num_perturbations):
        # Create perturbed image
        perturbed_image = _perturb_image(image, magnitude=perturbation_magnitude)
        
        # Get explanation for perturbed image
        perturbed_explanation = explainer.explain(perturbed_image)
        perturbed_saliency = perturbed_explanation['saliency_map']
        
        # Compute similarity between saliency maps
        similarity = _compute_saliency_similarity(original_saliency, perturbed_saliency)
        similarities.append(similarity)
    
    # Average similarity
    stability_score = np.mean(similarities)
    
    return stability_score, similarities


# Helper functions

def _get_confidence(model: nn.Module, image: torch.Tensor, target_class: int, device: torch.device) -> float:
    """Get model confidence for target class."""
    image = image.to(device)
    
    with torch.no_grad():
        # Use auxiliary model for efficiency
        y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = model.aux_model(image)
        probs = F.softmax(y0_aux_global, dim=1)
        confidence = probs[0, target_class].item()
    
    return confidence


def _get_sorted_pixel_coords(saliency_map: np.ndarray, descending: bool = True) -> list:
    """Get pixel coordinates sorted by saliency."""
    H, W = saliency_map.shape
    
    # Flatten and get indices
    flat_saliency = saliency_map.flatten()
    sorted_indices = np.argsort(flat_saliency)
    
    if descending:
        sorted_indices = sorted_indices[::-1]
    
    # Convert flat indices to (y, x) coordinates
    coords = []
    for idx in sorted_indices:
        y = idx // W
        x = idx % W
        coords.append((y, x))
    
    return coords


def _reveal_pixels(occluded_image: torch.Tensor, source_image: torch.Tensor, pixel_coords: list) -> torch.Tensor:
    """Reveal specified pixels from source image."""
    result = occluded_image.clone()
    
    for y, x in pixel_coords:
        result[0, :, y, x] = source_image[0, :, y, x]
    
    return result


def _reveal_regions(occluded_image: torch.Tensor, source_image: torch.Tensor, region_coords: list, region_size: int) -> torch.Tensor:
    """Reveal specified regions (blocks of pixels) from source image."""
    result = occluded_image.clone()
    
    for y, x in region_coords:
        y_start = y * region_size
        y_end = min((y + 1) * region_size, result.shape[2])
        x_start = x * region_size
        x_end = min((x + 1) * region_size, result.shape[3])
        result[0, :, y_start:y_end, x_start:x_end] = source_image[0, :, y_start:y_end, x_start:x_end]
    
    return result


def _occlude_pixels(image: torch.Tensor, pixel_coords: list, method: str = 'blur') -> torch.Tensor:
    """Occlude specified pixels in image."""
    result = image.clone()
    
    if method == 'blur':
        # Blur the entire image once, then copy blurred pixels
        blurred = _blur_image(image, sigma=10.0)
        for y, x in pixel_coords:
            result[0, :, y, x] = blurred[0, :, y, x]
    else:  # 'zero'
        for y, x in pixel_coords:
            result[0, :, y, x] = 0.0
    
    return result


def _occlude_regions(image: torch.Tensor, region_coords: list, region_size: int, method: str = 'blur') -> torch.Tensor:
    """Occlude specified regions (blocks of pixels) in image."""
    result = image.clone()
    
    if method == 'blur':
        blurred = _blur_image(image, sigma=10.0)
        for y, x in region_coords:
            y_start = y * region_size
            y_end = min((y + 1) * region_size, result.shape[2])
            x_start = x * region_size
            x_end = min((x + 1) * region_size, result.shape[3])
            result[0, :, y_start:y_end, x_start:x_end] = blurred[0, :, y_start:y_end, x_start:x_end]
    else:  # 'zero'
        for y, x in region_coords:
            y_start = y * region_size
            y_end = min((y + 1) * region_size, result.shape[2])
            x_start = x * region_size
            x_end = min((x + 1) * region_size, result.shape[3])
            result[0, :, y_start:y_end, x_start:x_end] = 0.0
    
    return result


def _blur_image(image: torch.Tensor, sigma: float = 10.0) -> torch.Tensor:
    """Apply Gaussian blur to image."""
    # Convert to numpy, blur, convert back
    img_np = image[0].cpu().numpy()  # (C, H, W)
    
    blurred_channels = []
    for c in range(img_np.shape[0]):
        blurred = gaussian_filter(img_np[c], sigma=sigma)
        blurred_channels.append(blurred)
    
    blurred_np = np.stack(blurred_channels, axis=0)
    blurred_tensor = torch.from_numpy(blurred_np).unsqueeze(0).to(image.device)
    
    return blurred_tensor


def _perturb_image(image: torch.Tensor, magnitude: float = 0.1) -> torch.Tensor:
    """Add random Gaussian noise to image."""
    noise = torch.randn_like(image) * magnitude
    perturbed = image + noise
    
    # Clip to valid range (assuming image is normalized)
    perturbed = torch.clamp(perturbed, 0, 1)
    
    return perturbed


def _compute_saliency_similarity(saliency1: np.ndarray, saliency2: np.ndarray) -> float:
    """
    Compute similarity between two saliency maps using Pearson correlation.
    
    Args:
        saliency1: First saliency map (H, W)
        saliency2: Second saliency map (H, W)
        
    Returns:
        Correlation coefficient in [-1, 1] (higher is more similar)
    """
    # Flatten
    s1_flat = saliency1.flatten()
    s2_flat = saliency2.flatten()
    
    # Compute Pearson correlation
    correlation = np.corrcoef(s1_flat, s2_flat)[0, 1]
    
    # Handle NaN (constant saliency maps)
    if np.isnan(correlation):
        correlation = 0.0
    
    return correlation


def compute_all_metrics(
    model: nn.Module,
    explainer: Any,
    image: torch.Tensor,
    saliency_map: np.ndarray,
    target_class: int,
    config: Dict[str, Any],
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Compute all metrics for a given saliency map.
    
    Args:
        model: The trained model
        explainer: XAI explainer instance
        image: Input image
        saliency_map: Saliency map to evaluate
        target_class: Target class
        config: Configuration dict with metric settings
        device: Device to run on
        
    Returns:
        Dictionary with all metric results
    """
    metrics_config = config.get('metrics', {})
    
    # Get precision mode
    precision_mode = metrics_config.get('precision_mode', 'balanced')
    
    # Get parameters
    insertion_steps = metrics_config.get('insertion_steps', 20)
    deletion_steps = metrics_config.get('deletion_steps', 20)
    stability_perturbations = metrics_config.get('stability_perturbations', 20)
    occlusion_method = metrics_config.get('occlusion_method', 'blur')
    
    # Compute metrics with precision mode support
    insertion_auc, ins_fractions, ins_confidences = compute_insertion_auc(
        model, image, saliency_map, target_class, 
        steps=insertion_steps, occlusion_method=occlusion_method, 
        device=device, precision_mode=precision_mode
    )
    
    deletion_auc, del_fractions, del_confidences = compute_deletion_auc(
        model, image, saliency_map, target_class,
        steps=deletion_steps, occlusion_method=occlusion_method, 
        device=device, precision_mode=precision_mode
    )
    
    stability, similarities = compute_stability(
        explainer, image, num_perturbations=stability_perturbations, device=device
    )
    
    return {
        'insertion_auc': insertion_auc,
        'insertion_fractions': ins_fractions,
        'insertion_confidences': ins_confidences,
        'deletion_auc': deletion_auc,
        'deletion_fractions': del_fractions,
        'deletion_confidences': del_confidences,
        'stability': stability,
        'stability_similarities': similarities,
    }
