# XAI Framework for Diffusion-Based Medical Image Classification

An explainability framework for interpreting **DiffMIC-v2** diffusion classifiers, designed for diabetic retinopathy grading.

## Why XAI for Diffusion Models?

Traditional XAI methods (Grad-CAM, SHAP) assume single-pass inference. Diffusion classifiers are fundamentally different - they refine predictions over 1000 iterative timesteps. This creates unique challenges:

- **Temporal Opacity**: Which timesteps actually matter for the decision?
- **Dual Conditioning**: How do global vs. local features influence the prediction?
- **Iterative Refinement**: Does the model "change its mind" during denoising?

This framework addresses these gaps with six purpose-built explainability techniques.

## DiffMIC-v2 Model

The underlying DiffMIC-v2 implementation can be found here:

**[DiffMIC-v2 Implementation Repository](https://github.com/garg-tejas/diffmic-from-scratch)**

## XAI Components

| Explainer                      | Purpose                                                                     |
| ------------------------------ | --------------------------------------------------------------------------- |
| **Attention Explainer**        | Visualizes DCG saliency maps and ROI attention weights                      |
| **Diffusion Explainer**        | Tracks class probabilities across all 1000 timesteps                        |
| **Spatio-Temporal Trajectory** | Shows coarse-to-fine attention shifts over time                             |
| **Conditional Attribution**    | Backpropagates through diffusion to quantify global vs. local contributions |
| **Faithfulness Validator**     | Insertion/deletion games with AUC metrics                                   |
| **Counterfactual Explainer**   | Generates minimal changes to flip predictions                               |

### Key Outputs

- Saliency maps with ROI overlays
- Probability trajectory plots
- Global vs. local attribution scores
- Faithfulness AUC curves
- Counterfactual comparisons

## Quick Start

```bash
git clone https://github.com/garg-tejas/xai-diffusion-based-models.git
cd xai-diffusion-based-models
pip install -r requirements.txt
python main.py
```

### Configuration

Key settings in `config/xai_config.yaml`:

```yaml
sampling:
  samples_per_class: 10
  selection_strategy: "balanced" # 'balanced', 'correct_only', 'incorrect_only'

explainers:
  attention:
    enabled: true
  diffusion:
    enabled: true
  faithfulness:
    enabled: true
    deletion_steps: 20
  conditional_attribution:
    enabled: true
  counterfactual:
    enabled: true
    guidance_scale: 5.0
```

## Results

On the APTOS 2019 diabetic retinopathy dataset:

| Metric       | Value         |
| ------------ | ------------- |
| **Accuracy** | 84.1%         |
| **F1-Score** | 69.8% (macro) |

## Project Structure

```
├── config/
│   └── xai_config.yaml          # Pipeline configuration
├── core/
│   ├── base_explainer.py        # Abstract explainer class
│   ├── model_loader.py          # Checkpoint loading
│   └── sample_selector.py       # Stratified sample selection
├── explainers/
│   ├── attention_explainer.py
│   ├── diffusion_explainer.py
│   ├── spatiotemporal_trajectory_explainer.py
│   ├── conditional_attribution_explainer.py
│   ├── faithfulness_validator.py
│   └── counterfactual_explainer.py
├── visualizers/
│   ├── attention_vis.py
│   ├── trajectory_vis.py
│   ├── spatiotemporal_vis.py
│   ├── attribution_vis.py
│   ├── faithfulness_vis.py
│   └── counterfactual_vis.py
└── utils/
    ├── image_utils.py
    └── inference.py
```

## Limitations

- **Compute Cost**: Attribution requires backprop through 1000 timesteps
- **2D Only**: Designed for 2D fundus images, not 3D volumes
- **Clinical Validation**: Needs ophthalmologist studies

---

> Built for DiffMIC-v2: _Medical Image Classification via Dual-granularity Conditional Guidance Diffusion_
