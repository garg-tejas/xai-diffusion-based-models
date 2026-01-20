"""
Explainer modules for different XAI techniques.

Current implementations:
- attention_explainer: Extract built-in attention mechanisms
- diffusion_explainer: Track diffusion denoising trajectory
- gradient_explainer: Grad-CAM, Integrated Gradients, Score-CAM
- conditional_attribution_explainer: Backprop through diffusion
- counterfactual_explainer: Generate minimal changes to flip predictions
"""

from .attention_explainer import AttentionExplainer
from .diffusion_explainer import DiffusionExplainer
from .guidance_explainer import GuidanceMapExplainer
from .feature_prior_explainer import FeaturePriorExplainer
from .noise_explainer import NoiseExplainer
from .faithfulness_validator import FaithfulnessValidator
from .conditional_attribution_explainer import ConditionalAttributionExplainer
from .spatiotemporal_trajectory_explainer import SpatioTemporalTrajectoryExplainer
from .counterfactual_explainer import GenerativeCounterfactualExplainer

# New gradient-based explainers
from .gradcam_explainer import GradCAMExplainer
from .integrated_gradients_explainer import IntegratedGradientsExplainer
from .scorecam_explainer import ScoreCAMExplainer

__all__ = [
    'AttentionExplainer',
    'DiffusionExplainer',
    'GuidanceMapExplainer',
    'FeaturePriorExplainer',
    'NoiseExplainer',
    'FaithfulnessValidator',
    'ConditionalAttributionExplainer',
    'SpatioTemporalTrajectoryExplainer',
    'GenerativeCounterfactualExplainer',
    'GradCAMExplainer',
    'IntegratedGradientsExplainer',
    'ScoreCAMExplainer',
]
