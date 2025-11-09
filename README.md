# XAI Framework for Diffusion-Based Classification

Explainable AI framework for diffusion-based medical image classifiers.

## Usage

```bash
# Run with default config
python main.py

# Custom config
python main.py --config config/xai_config.yaml

# Specify checkpoint and sample count
python main.py --checkpoint logs/aptos/version_0/checkpoints/last.ckpt --samples 5
```

## Configuration

Edit `config/xai_config.yaml` to configure:

- Model checkpoint path
- Sample selection (count, strategy)
- Enabled explainers
- Visualization settings
- Output formats

## Files

### Core

- `main.py` - Main pipeline entry point, coordinates all components
- `core/base_explainer.py` - Abstract base class for all explainers
- `core/model_loader.py` - Loads trained model from PyTorch Lightning checkpoint
- `core/sample_selector.py` - Selects representative samples from CSV predictions

### Explainers

- `explainers/attention_explainer.py` - Extracts attention from auxiliary DCG model (saliency maps, patch locations, attention weights)
- `explainers/diffusion_explainer.py` - Tracks diffusion denoising trajectory (probability evolution, prediction changes)
- `explainers/guidance_explainer.py` - Extracts dense guidance maps showing interpolation between global and local priors
- `explainers/feature_prior_explainer.py` - Analyzes feature fusion from Transformer and CNN encoders
- `explainers/noise_explainer.py` - Analyzes heterologous noise patterns during diffusion

### Visualizers

- `visualizers/attention_vis.py` - Creates attention visualizations (heatmaps, patch highlighting, animations)
- `visualizers/trajectory_vis.py` - Creates diffusion trajectory plots (probability evolution, confidence, entropy)
- `visualizers/guidance_vis.py` - Creates guidance map visualizations (heatmaps, interpolation plots, animations)
- `visualizers/feature_prior_vis.py` - Creates feature prior visualizations (contributions, fusion weights, feature space)
- `visualizers/noise_vis.py` - Creates noise analysis visualizations (interaction maps, magnitude plots, animations)
- `visualizers/combined_visualizer.py` - Creates synchronized multi-panel animations
- `visualizers/report_generator.py` - Generates HTML reports with embedded visualizations

### Utils

- `utils/image_utils.py` - Image processing utilities (loading, preprocessing, heatmap overlays, bounding boxes)
- `utils/animation_utils.py` - Animation helpers (frame capture, GIF creation)

### Config

- `config/xai_config.yaml` - Configuration file (model, sampling, explainers, visualization, output settings)

## Outputs

Results are saved in `outputs/`:

- `images/` - PNG visualizations and GIF animations
- `html/` - HTML reports
- `arrays/` - NumPy arrays (raw data)
- `run_config.yaml` - Copy of configuration used
- `xai_processing.log` - Processing log

## Output Files

### Images

- `class_X_sample_Y_attention.png` - Attention visualization (4-panel)
- `class_X_sample_Y_trajectory.png` - Diffusion trajectory (4-panel)
- `class_X_sample_Y_guidance_heatmap.png` - Guidance map heatmap
- `class_X_sample_Y_prior_interpolation.png` - Prior interpolation plot
- `class_X_sample_Y_feature_contributions.png` - Feature contribution plot
- `class_X_sample_Y_noise_interaction.png` - Noise interaction map
- `class_X_sample_Y_attention_evolution.gif` - Attention evolution animation
- `class_X_sample_Y_denoising.gif` - Denoising process animation
- `class_X_sample_Y_guidance_evolution.gif` - Guidance evolution animation
- `class_X_sample_Y_feature_evolution.gif` - Feature evolution animation
- `class_X_sample_Y_noise_evolution.gif` - Noise evolution animation
- `class_X_sample_Y_synchronized.gif` - Synchronized multi-panel animation
- `class_X_sample_Y_denoising_sequence.gif` - Image denoising sequence

### HTML Reports

- `class_X_sample_Y_report.html` - Individual sample report
- `index.html` - Summary report across all samples

### Arrays

- `class_X_sample_Y_data.npz` - Raw explanation data (NumPy archive)
