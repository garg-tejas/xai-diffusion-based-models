"""
Report Generator Module

This module creates comprehensive HTML reports combining all XAI explanations.
The reports are interactive, professional, and suitable for research or clinical use.

Features:
- Embeds visualizations (images, plots, tables)
- Interactive elements (collapsible sections, hover effects)
- Summary statistics across multiple samples
- Comparison views
- Export options

Purpose:
- Present complete XAI analysis in one document
- Enable easy sharing and archiving
- Support clinical validation studies
- Facilitate model debugging and research

Future Extensions:
- JavaScript interactivity (zoom, pan, compare)
- Export to PDF
- Integration with medical imaging viewers
- Collaborative annotation features
"""

import sys
from pathlib import Path
import base64
from io import BytesIO
from typing import Dict, Any, List, Optional
from PIL import Image
import numpy as np
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ReportGenerator:
    """
    Generate comprehensive HTML reports for XAI analysis.
    
    This class creates professional, interactive HTML reports that combine
    all explana visualizations, explanations, and metadata into a single document.
    
    Features:
    - Clean, responsive design
    - Embedded visualizations
    - Collapsible sections for organization
    - Summary statistics
    - Export-ready format
    
    Usage:
        >>> generator = ReportGenerator(config)
        >>> generator.create_sample_report(
        ...     image, attention_exp, diffusion_exp, visualizations
        ... )
        >>> generator.create_summary_report(all_samples)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the report generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.class_names = config.get('class_names', {})
        
        print(f"[ReportGenerator] Initialized")
    
    @staticmethod
    def _to_float(value) -> float:
        """Convert value to Python float, handling numpy arrays/scalars."""
        arr = np.asarray(value)
        return float(arr.item() if arr.size == 1 else arr.flatten()[0])
    
    @staticmethod
    def _to_int(value) -> int:
        """Convert value to Python int, handling numpy arrays/scalars."""
        arr = np.asarray(value)
        return int(arr.item() if arr.size == 1 else arr.flatten()[0])
    
    def create_sample_report(self,
                            sample_info: Dict[str, Any],
                            attention_exp: Dict[str, Any],
                            diffusion_exp: Dict[str, Any],
                            visualizations: Dict[str, np.ndarray],
                            save_path: str,
                            animation_paths: Optional[Dict[str, str]] = None) -> None:
        """
        Create HTML report for a single sample.
        
        Args:
            sample_info: Sample metadata (path, label, etc.)
            attention_exp: Explanation from AttentionExplainer
            diffusion_exp: Explanation from DiffusionExplainer
            visualizations: Dictionary of visualization arrays
            save_path: Path to save HTML file
            animation_paths: Optional dictionary with animation GIF paths:
                - 'denoising': Probability bars animation
                - 'attention_evolution': Attention overlay animation
                - 'synchronized': Combined multi-panel animation
                - 'denoising_sequence': Image with attention evolution
        """
        html_content = self._generate_html_header()
        
        # Title and metadata
        html_content += self._generate_sample_header(sample_info, attention_exp, diffusion_exp)
        
        # Animations section (if available)
        if animation_paths:
            html_content += self._generate_animations_section(animation_paths, visualizations)
        
        # Attention section
        html_content += self._generate_attention_section(attention_exp, visualizations)
        
        # Diffusion section
        html_content += self._generate_diffusion_section(diffusion_exp, visualizations)
        
        # Analysis section
        html_content += self._generate_analysis_section(attention_exp, diffusion_exp, sample_info)
        
        html_content += self._generate_html_footer()
        
        # Save
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[ReportGenerator] Saved report to {save_path}")
    
    def create_summary_report(self,
                             all_results: List[Dict[str, Any]],
                             save_path: str) -> None:
        """
        Create summary HTML report across all samples.
        
        Args:
            all_results: List of results for each sample
            save_path: Path to save HTML file
        """
        html_content = self._generate_html_header()
        
        # Title
        html_content += f"""
        <div class="container">
            <h1>XAI Analysis Summary Report</h1>
            <p class="subtitle">Comprehensive analysis of {len(all_results)} samples</p>
            <hr>
        """
        
        # Overall statistics
        html_content += self._generate_summary_statistics(all_results)
        
        # Individual sample links
        html_content += self._generate_sample_list(all_results)
        
        html_content += "</div>"
        html_content += self._generate_html_footer()
        
        # Save
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[ReportGenerator] Saved summary report to {save_path}")
    
    def _create_animation_embed(self,
                               static_image_path: Optional[str],
                               animated_gif_path: str,
                               caption: str,
                               element_id: str) -> str:
        """
        Create HTML for animation with static/animated toggle.
        
        Args:
            static_image_path: Path to static image (PNG), or None
            animated_gif_path: Path to animated GIF
            caption: Caption text for the animation
            element_id: Unique ID for this animation element
        
        Returns:
            HTML string with embedded animation and controls
        
        Features:
            - Shows static image by default (if available)
            - Toggle button to switch to animated GIF
            - Play/pause functionality
        """
        # Encode paths for HTML
        static_path = static_image_path if static_image_path else animated_gif_path
        
        html = f'''
        <div class="animation-container" id="container_{element_id}">
            <div class="animation-caption">{caption}</div>
            <div class="animation-wrapper">
                <img id="static_{element_id}" 
                     src="../{static_path}" 
                     style="display:block; max-width:100%; height:auto;"
                     alt="{caption} (static)">
                <img id="animated_{element_id}" 
                     src="../{animated_gif_path}" 
                     style="display:none; max-width:100%; height:auto;"
                     alt="{caption} (animated)">
            </div>
            <div class="animation-controls">
                <button onclick="toggleAnimation('{element_id}')" 
                        id="btn_{element_id}"
                        class="control-button">
                    &#9654; Play Animation
                </button>
                <span class="control-info" id="info_{element_id}">
                    (Showing static image)
                </span>
            </div>
        </div>
        '''
        return html
    
    def _generate_animations_section(self, 
                                    animation_paths: Dict[str, str],
                                    visualizations: Dict[str, np.ndarray]) -> str:
        """
        Generate HTML section for animations.
        
        Args:
            animation_paths: Dictionary of animation GIF paths
            visualizations: Dictionary of static visualizations
        
        Returns:
            HTML string with animations section
        """
        html = '''
        <h2>Animations</h2>
        <p>Interactive animations showing the evolution of predictions, attention, and model behavior through the denoising process.</p>
        <div class="visualizations-grid">
        '''
        
        # Synchronized animation
        if animation_paths.get('synchronized'):
            html += self._create_animation_embed(
                static_image_path=None,
                animated_gif_path=animation_paths['synchronized'],
                caption="Synchronized Multi-Panel View: Shows image with attention, probability bars, metadata, and model comparison evolving together.",
                element_id="sync"
            )
        
        # Image denoising sequence
        if animation_paths.get('denoising_sequence'):
            html += self._create_animation_embed(
                static_image_path=None,
                animated_gif_path=animation_paths['denoising_sequence'],
                caption="Image Denoising Sequence: Original image with attention overlay evolving through timesteps.",
                element_id="img_seq"
            )
        
        # Probability bars animation
        if animation_paths.get('denoising'):
            html += self._create_animation_embed(
                static_image_path=None,
                animated_gif_path=animation_paths['denoising'],
                caption="Probability Evolution: Bar chart showing how class probabilities change through denoising steps.",
                element_id="probs"
            )
        
        # Attention evolution animation
        if animation_paths.get('attention_evolution'):
            html += self._create_animation_embed(
                static_image_path=None,
                animated_gif_path=animation_paths['attention_evolution'],
                caption="Attention Evolution: How attention regions change as the model refines its prediction.",
                element_id="attn"
            )
        
        html += '''
        </div>
        <div class="section-separator"></div>
        '''
        return html
    
    def _generate_html_header(self) -> str:
        """Generate HTML header with CSS styling."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XAI Analysis Report</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }
        
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        
        h2 {
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
            padding-left: 10px;
            border-left: 4px solid #3498db;
        }
        
        h3 {
            color: #555;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: #7f8c8d;
            font-size: 1.1em;
            margin-bottom: 10px;
        }
        
        .metadata {
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }
        
        .metadata-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 10px;
        }
        
        .metadata-item {
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }
        
        .metadata-label {
            font-weight: bold;
            color: #555;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .metadata-value {
            font-size: 1.2em;
            color: #2c3e50;
            margin-top: 5px;
        }
        
        .status-correct {
            color: #27ae60;
            font-weight: bold;
        }
        
        .status-incorrect {
            color: #e74c3c;
            font-weight: bold;
        }
        
        .section {
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #fafafa;
        }
        
        .visualization {
            width: 100%;
            max-width: 100%;
            height: auto;
            margin: 15px 0;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .vis-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }
        
        tr:hover {
            background-color: #f5f5f5;
        }
        
        .collapsible {
            background-color: #3498db;
            color: white;
            cursor: pointer;
            padding: 15px;
            width: 100%;
            border: none;
            text-align: left;
            outline: none;
            font-size: 1.1em;
            font-weight: bold;
            border-radius: 5px;
            margin-top: 10px;
        }
        
        .collapsible:hover {
            background-color: #2980b9;
        }
        
        .collapsible-content {
            padding: 20px;
            display: none;
            overflow: hidden;
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-top: none;
            border-radius: 0 0 5px 5px;
        }
        
        .highlight-box {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
            border-radius: 3px;
        }
        
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ddd;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }
        
        .animation-container {
            margin: 20px 0;
            padding: 15px;
            background: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        
        .animation-caption {
            font-weight: bold;
            margin-bottom: 10px;
            color: #2c3e50;
        }
        
        .animation-wrapper {
            position: relative;
            margin-bottom: 10px;
        }
        
        .animation-controls {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .control-button {
            padding: 8px 16px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
        }
        
        .control-button:hover {
            background: #2980b9;
        }
        
        .control-info {
            color: #666;
            font-size: 13px;
        }
    </style>
    <script>
        function toggleCollapsible(element) {
            element.classList.toggle('active');
            var content = element.nextElementSibling;
            if (content.style.display === 'block') {
                content.style.display = 'none';
            } else {
                content.style.display = 'block';
            }
        }
    </script>
</head>
<body>
"""
    
    def _generate_sample_header(self, sample_info, attention_exp, diffusion_exp) -> str:
        """Generate header section with sample metadata."""
        img_path = sample_info.get('img_path', 'Unknown')
        gt_label = sample_info.get('ground_truth', -1)
        
        att_pred = attention_exp['prediction']
        att_conf = attention_exp['confidence']
        diff_pred = diffusion_exp['prediction']
        diff_conf = diffusion_exp['confidence']
        
        # Convert numpy scalars to Python types
        att_conf = self._to_float(att_conf)
        diff_conf = self._to_float(diff_conf)
        
        gt_name = self.class_names.get(str(gt_label), f"Class {gt_label}") if gt_label != -1 else "Unknown"
        att_name = self.class_names.get(str(att_pred), f"Class {att_pred}")
        diff_name = self.class_names.get(str(diff_pred), f"Class {diff_pred}")
        
        att_correct = (att_pred == gt_label) if gt_label != -1 else None
        diff_correct = (diff_pred == gt_label) if gt_label != -1 else None
        
        html = f"""
<div class="container">
    <h1>XAI Analysis Report</h1>
    <p class="subtitle">Sample: {Path(img_path).name}</p>
    <hr>
    
    <div class="metadata">
        <h3>Sample Information</h3>
        <div class="metadata-grid">
            <div class="metadata-item">
                <div class="metadata-label">Image Path</div>
                <div class="metadata-value">{img_path}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Ground Truth</div>
                <div class="metadata-value">{gt_name}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Auxiliary Prediction</div>
                <div class="metadata-value {'status-correct' if att_correct else 'status-incorrect' if att_correct is not None else ''}">{att_name} ({att_conf:.3f})</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Diffusion Prediction</div>
                <div class="metadata-value {'status-correct' if diff_correct else 'status-incorrect' if diff_correct is not None else ''}">{diff_name} ({diff_conf:.3f})</div>
            </div>
        </div>
    </div>
"""
        return html
    
    def _generate_attention_section(self, attention_exp, visualizations) -> str:
        """Generate attention explanation section."""
        html = """
    <h2>Attention-Based Explanations</h2>
    <div class="section">
        <p>Explanation from the auxiliary DCG model's built-in attention mechanisms.</p>
"""
        
        # Add visualizations
        if 'attention_overview' in visualizations:
            img_data = self._array_to_base64(visualizations['attention_overview'])
            html += f'<img src="data:image/png;base64,{img_data}" class="visualization" alt="Attention Overview">'
        
        # Collapsible details
        patch_attns = attention_exp.get('patch_attention', [])
        html += """
        <button class="collapsible" onclick="toggleCollapsible(this)">▶ Detailed Attention Data</button>
        <div class="collapsible-content">
"""
        
        # Patch attention table
        if len(patch_attns) > 0:
            # Ensure patch_attention is 1D array
            patch_attns = np.asarray(patch_attns).flatten()
            
            html += "<h4>Patch Attention Weights</h4><table><tr><th>Patch</th><th>Attention Weight</th></tr>"
            for i, weight in enumerate(patch_attns):
                weight_val = float(weight)
                html += f"<tr><td>Patch {i}</td><td>{weight_val:.4f}</td></tr>"
            html += "</table>"
        
        html += "</div></div>"
        return html
    
    def _generate_diffusion_section(self, diffusion_exp, visualizations) -> str:
        """Generate diffusion trajectory section."""
        stability = diffusion_exp.get('stability_score', 0)
        convergence = diffusion_exp.get('convergence_step', 0)
        
        # Convert numpy scalars to Python types
        stability = self._to_float(stability)
        convergence = self._to_int(convergence)
        
        html = f"""
    <h2>Diffusion Trajectory Analysis</h2>
    <div class="section">
        <p>Analysis of how predictions evolve through the diffusion denoising process.</p>
        
        <div class="highlight-box">
            <strong>Stability Score:</strong> {stability:.3f} | 
            <strong>Convergence Timestep:</strong> {convergence}
        </div>
"""
        
        # Add visualizations
        if 'trajectory' in visualizations:
            img_data = self._array_to_base64(visualizations['trajectory'])
            html += f'<img src="data:image/png;base64,{img_data}" class="visualization" alt="Diffusion Trajectory">'
        
        html += "</div>"
        return html
    
    def _generate_analysis_section(self, attention_exp, diffusion_exp, sample_info) -> str:
        """Generate analysis and insights section."""
        gt_label = sample_info.get('ground_truth', -1)
        
        html = """
    <h2>Analysis & Insights</h2>
    <div class="section">
"""
        
        # Model agreement
        if attention_exp['prediction'] == diffusion_exp['prediction']:
            html += '<div class="highlight-box">[OK] <strong>Strong Agreement:</strong> Both models predict the same class.</div>'
        else:
            html += '<div class="highlight-box">[WARNING] <strong>Disagreement:</strong> Models predict different classes.</div>'
        
        # Correctness if GT available
        if gt_label != -1:
            att_correct = attention_exp['prediction'] == gt_label
            diff_correct = diffusion_exp['prediction'] == gt_label
            
            if att_correct and diff_correct:
                html += '<p><strong>[OK] Both models predict correctly</strong></p>'
            elif diff_correct and not att_correct:
                html += '<p><strong>Diffusion model corrects auxiliary error</strong></p>'
            elif att_correct and not diff_correct:
                html += '<p><strong>[WARNING] Diffusion introduces error</strong></p>'
            else:
                html += '<p><strong>[ERROR] Both models predict incorrectly</strong></p>'
        
        html += "</div>"
        return html
    
    def _generate_summary_statistics(self, all_results) -> str:
        """Generate summary statistics across all samples."""
        total = len(all_results)
        
        # Count correct/incorrect
        att_correct = sum(1 for r in all_results if r.get('attention_correct', False))
        diff_correct = sum(1 for r in all_results if r.get('diffusion_correct', False))
        
        # Average confidence
        avg_att_conf = float(np.mean([r.get('attention_confidence', 0) for r in all_results]))
        avg_diff_conf = float(np.mean([r.get('diffusion_confidence', 0) for r in all_results]))
        
        html = f"""
    <div class="metadata">
        <h3>Overall Statistics</h3>
        <div class="metadata-grid">
            <div class="metadata-item">
                <div class="metadata-label">Total Samples</div>
                <div class="metadata-value">{total}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Auxiliary Accuracy</div>
                <div class="metadata-value">{att_correct}/{total} ({att_correct/total*100:.1f}%)</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Diffusion Accuracy</div>
                <div class="metadata-value">{diff_correct}/{total} ({diff_correct/total*100:.1f}%)</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Avg. Confidence</div>
                <div class="metadata-value">Aux: {avg_att_conf:.3f}, Diff: {avg_diff_conf:.3f}</div>
            </div>
        </div>
    </div>
"""
        return html
    
    def _generate_sample_list(self, all_results) -> str:
        """Generate list of analyzed samples."""
        html = "<h2>Analyzed Samples</h2><table><tr><th>Sample</th><th>Ground Truth</th><th>Aux Pred</th><th>Diff Pred</th><th>Report</th></tr>"
        
        for r in all_results:
            sample_name = Path(r.get('img_path', 'Unknown')).name
            gt_val = r.get('ground_truth', -1)
            gt = self.class_names.get(str(gt_val), 'Unknown') if gt_val != -1 else 'Unknown'
            att_pred_val = r.get('attention_prediction', -1)
            att_pred = self.class_names.get(str(att_pred_val), 'Unknown') if att_pred_val != -1 else 'Unknown'
            diff_pred_val = r.get('diffusion_prediction', -1)
            diff_pred = self.class_names.get(str(diff_pred_val), 'Unknown') if diff_pred_val != -1 else 'Unknown'
            report_link = r.get('report_path', '#')
            
            html += f"<tr><td>{sample_name}</td><td>{gt}</td><td>{att_pred}</td><td>{diff_pred}</td><td><a href='{report_link}'>View Report</a></td></tr>"
        
        html += "</table>"
        return html
    
    def _generate_html_footer(self) -> str:
        """Generate HTML footer with JavaScript for animation controls."""
        return """
    <div class="footer">
        <p>Generated by XAI Framework for Diffusion-Based Classification</p>
        <p>© 2025 | Research Tool</p>
    </div>
</div>
<script>
    // Animation toggle functionality
    function toggleAnimation(elementId) {
        const staticImg = document.getElementById('static_' + elementId);
        const animatedImg = document.getElementById('animated_' + elementId);
        const btn = document.getElementById('btn_' + elementId);
        const info = document.getElementById('info_' + elementId);
        
        if (staticImg.style.display === 'block') {
            // Switch to animated
            staticImg.style.display = 'none';
            animatedImg.style.display = 'block';
            btn.innerHTML = '&#9632; Stop Animation';
            info.textContent = '(Playing animation)';
        } else {
            // Switch to static
            staticImg.style.display = 'block';
            animatedImg.style.display = 'none';
            btn.innerHTML = '&#9654; Play Animation';
            info.textContent = '(Showing static image)';
        }
    }
</script>
</body>
</html>
"""
    
    def _array_to_base64(self, array: np.ndarray) -> str:
        """Convert numpy array to base64 string for embedding."""
        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = (array * 255).astype(np.uint8)
            else:
                array = array.astype(np.uint8)
        
        img = Image.fromarray(array)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str


"""
Usage Example:

from xai.visualizers.report_generator import ReportGenerator

# Create generator
config = {
    'class_names': {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}
}
generator = ReportGenerator(config)

# Generate single sample report
generator.create_sample_report(
    sample_info={'img_path': 'image.png', 'ground_truth': 2},
    attention_exp=attention_explanation,
    diffusion_exp=diffusion_explanation,
    visualizations={
        'attention_overview': attention_vis,
        'trajectory': trajectory_vis
    },
    save_path='outputs/html/sample_report.html'
)

# Generate summary report
all_results = [...]  # List of all sample results
generator.create_summary_report(
    all_results,
    save_path='outputs/html/index.html'
)
"""

