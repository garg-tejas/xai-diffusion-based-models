"""
Trajectory Visualization Module

This module creates visualizations of the diffusion denoising trajectory.
It shows how predictions evolve from noise to the final classification.

Visualization Types:
1. Time-series plot: Class probabilities vs timestep
2. Confidence evolution: How confident the model becomes over time
3. Entropy evolution: Uncertainty reduction through denoising
4. Prediction stability: Track when prediction changes
5. Comparison: Diffusion vs auxiliary predictions

Purpose:
- Understand diffusion's "reasoning process"
- Identify critical timesteps
- Validate prediction stability
- Debug convergence issues

Future Extensions:
- Animation (GIF/MP4) of denoising process
- 3D trajectory in probability simplex
- Multi-sample trajectory clustering
- Counterfactual trajectory comparison
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import seaborn as sns
from typing import Dict, Any, Optional, List
from PIL import Image
import imageio

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from xai.utils.image_utils import save_visualization


class TrajectoryVisualizer:
    """
    Visualize diffusion denoising trajectories.
    
    This class takes the output from DiffusionExplainer and creates
    visualizations that show how predictions evolve through timesteps.
    
    Key visualizations:
    - Probability trajectories for all classes
    - Confidence and entropy evolution
    - Prediction stability analysis
    - Comparison with auxiliary model
    
    Attributes:
        config: Configuration dictionary
        class_names: Human-readable class names
        
    Usage:
        >>> visualizer = TrajectoryVisualizer(config)
        >>> visualizer.visualize_trajectory(
        ...     explanation, save_path='trajectory.png'
        ... )
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trajectory visualizer.
        
        Args:
            config: Configuration with settings:
                - figure_size: Default figure size
                - dpi: Image resolution
                - class_names: Dictionary mapping class IDs to names
        """
        self.config = config
        self.figsize = tuple(config.get('figure_size', [15, 10]))
        self.dpi = config.get('image_dpi', 300)
        self.class_names = config.get('class_names', {})
        
        print(f"[TrajectoryVisualizer] Initialized")
    
    def visualize_trajectory(self,
                            explanation: Dict[str, Any],
                            save_path: Optional[str] = None) -> np.ndarray:
        """
        Create comprehensive trajectory visualization.
        
        Args:
            explanation: Explanation dict from DiffusionExplainer
            save_path: Path to save visualization
        
        Returns:
            Visualization as numpy array
        
        Layout:
            Top row: Probability trajectories | Confidence & Entropy
            Bottom row: Prediction changes | Comparison with aux
        """
        # Extract data
        trajectory = explanation['trajectory']
        timesteps = explanation['timesteps']
        prediction = explanation['prediction']
        ground_truth = explanation.get('ground_truth', -1)
        
        # Create figure
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Probability trajectories
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_probability_trajectories(ax1, trajectory, timesteps, prediction, ground_truth)
        
        # 2. Confidence and entropy
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_confidence_entropy(ax2, explanation)
        
        # 3. Prediction changes
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_prediction_changes(ax3, trajectory, timesteps)
        
        # 4. Comparison with auxiliary
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_aux_comparison(ax4, explanation)
        
        # Overall title
        class_name = self.class_names.get(str(prediction), f"Class {prediction}")
        confidence = explanation['confidence']
        stability = explanation['stability_score']
        
        title = (f'Diffusion Trajectory Analysis\n'
                f'Prediction: {class_name} (confidence: {confidence:.3f}, '
                f'stability: {stability:.3f})')
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        # Convert to array - use tight_layout with rect parameter to avoid warnings
        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08, wspace=0.3)
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        
        buf = fig.canvas.buffer_rgba()
        vis_array = np.asarray(buf)[:, :, :3]
        
        plt.close(fig)
        
        if save_path:
            save_visualization(vis_array, save_path, self.dpi)
        
        return vis_array
    
    def _plot_probability_trajectories(self, ax, trajectory, timesteps, prediction, ground_truth):
        """Plot class probability evolution over timesteps."""
        # Extract probabilities for each class
        probs_over_time = np.array([step['probs'] for step in trajectory])
        num_classes = probs_over_time.shape[1]
        
        # Plot each class
        for class_id in range(num_classes):
            class_name = self.class_names.get(str(class_id), f"C{class_id}")
            probs = probs_over_time[:, class_id]
            
            # Highlight predicted and ground truth classes
            if class_id == prediction:
                ax.plot(timesteps, probs, linewidth=3, label=f'{class_name} (predicted)',
                       linestyle='-', marker='o', markersize=4)
            elif class_id == ground_truth and ground_truth != -1:
                ax.plot(timesteps, probs, linewidth=2.5, label=f'{class_name} (ground truth)',
                       linestyle='--', marker='s', markersize=4)
            else:
                ax.plot(timesteps, probs, linewidth=1.5, label=class_name, alpha=0.6)
        
        ax.set_xlabel('Timestep (T → 0)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Class Probability', fontsize=12, fontweight='bold')
        ax.set_title('Probability Evolution', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(max(timesteps), min(timesteps))  # Reverse x-axis (T -> 0)
    
    def _plot_confidence_entropy(self, ax, explanation):
        """Plot confidence and entropy evolution."""
        timesteps = explanation['timesteps']
        confidence_traj = explanation['confidence_trajectory']
        entropy_traj = explanation['entropy_trajectory']
        convergence_step = explanation['convergence_step']
        
        # Create twin axis for entropy
        ax2 = ax.twinx()
        
        # Plot confidence
        line1 = ax.plot(timesteps, confidence_traj, 'b-', linewidth=2,
                       label='Confidence', marker='o', markersize=3)
        ax.set_xlabel('Timestep (T → 0)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Confidence (Max Prob)', fontsize=12, fontweight='bold', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        
        # Plot entropy
        line2 = ax2.plot(timesteps, entropy_traj, 'r--', linewidth=2,
                        label='Entropy', marker='s', markersize=3)
        ax2.set_ylabel('Entropy', fontsize=12, fontweight='bold', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Mark convergence point
        if convergence_step > 0:
            ax.axvline(x=convergence_step, color='green', linestyle=':', linewidth=2,
                      label=f'Convergence (t={convergence_step})')
        
        # Legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        if convergence_step > 0:
            lines.append(plt.Line2D([0], [0], color='green', linestyle=':', linewidth=2))
            labels.append(f'Convergence')
        ax.legend(lines, labels, loc='best', fontsize=9)
        
        ax.set_title('Confidence & Entropy Evolution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(max(timesteps), min(timesteps))
    
    def _plot_prediction_changes(self, ax, trajectory, timesteps):
        """Plot when predictions change."""
        # Extract predictions over time
        probs_over_time = np.array([step['probs'] for step in trajectory])
        predictions_over_time = np.argmax(probs_over_time, axis=1)
        
        # Find change points
        changes = []
        for i in range(1, len(predictions_over_time)):
            if predictions_over_time[i] != predictions_over_time[i-1]:
                changes.append((timesteps[i], predictions_over_time[i-1], predictions_over_time[i]))
        
        # Plot prediction over time
        ax.plot(timesteps, predictions_over_time, 'o-', linewidth=2, markersize=6,
               color='steelblue', label='Predicted Class')
        
        # Mark change points
        for t, old_pred, new_pred in changes:
            ax.axvline(x=t, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
            ax.text(t, old_pred + 0.1, f'→{new_pred}', fontsize=8, color='red',
                   ha='center', va='bottom')
        
        ax.set_xlabel('Timestep (T → 0)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Class', fontsize=12, fontweight='bold')
        ax.set_title(f'Prediction Changes (n={len(changes)})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(max(timesteps), min(timesteps))
        
        # Set y-ticks to class labels
        num_classes = int(predictions_over_time.max()) + 1
        ax.set_yticks(range(num_classes))
        ax.set_yticklabels([self.class_names.get(str(i), f"C{i}") for i in range(num_classes)],
                          fontsize=9)
    
    def _plot_aux_comparison(self, ax, explanation):
        """Compare diffusion vs auxiliary predictions."""
        # Get predictions
        diff_pred = explanation['prediction']
        diff_conf = explanation['confidence']
        aux_pred = explanation['aux_prediction']
        aux_conf = explanation['aux_confidence']
        ground_truth = explanation.get('ground_truth', -1)
        
        # Create bar chart
        categories = ['Auxiliary\nModel', 'Diffusion\nModel']
        predictions = [aux_pred, diff_pred]
        confidences = [aux_conf, diff_conf]
        
        x = np.arange(len(categories))
        
        # Color by correctness if ground truth available
        if ground_truth != -1:
            colors = ['green' if p == ground_truth else 'red' for p in predictions]
        else:
            colors = ['steelblue', 'coral']
        
        # Plot bars
        bars = ax.bar(x, confidences, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add prediction labels on bars
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            class_name = self.class_names.get(str(pred), f"C{pred}")
            ax.text(i, conf + 0.02, f'{class_name}\n({conf:.3f})',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Confidence', fontsize=12, fontweight='bold')
        ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add legend if ground truth available
        if ground_truth != -1:
            gt_name = self.class_names.get(str(ground_truth), f"C{ground_truth}")
            green_patch = mpatches.Patch(color='green', alpha=0.7, label='Correct')
            red_patch = mpatches.Patch(color='red', alpha=0.7, label='Incorrect')
            ax.legend(handles=[green_patch, red_patch], loc='upper right')
            ax.text(0.5, 0.95, f'Ground Truth: {gt_name}', transform=ax.transAxes,
                   ha='center', va='top', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    def visualize_probability_heatmap(self,
                                     explanation: Dict[str, Any],
                                     save_path: Optional[str] = None) -> np.ndarray:
        """
        Create heatmap showing probability evolution for all classes.
        
        Args:
            explanation: Explanation dict
            save_path: Save path
        
        Returns:
            Visualization array
        """
        trajectory = explanation['trajectory']
        timesteps = explanation['timesteps']
        
        # Extract probabilities
        probs_over_time = np.array([step['probs'] for step in trajectory])
        num_classes = probs_over_time.shape[1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create heatmap
        im = ax.imshow(probs_over_time.T, aspect='auto', cmap='YlOrRd',
                      interpolation='nearest', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_yticks(range(num_classes))
        ax.set_yticklabels([self.class_names.get(str(i), f"C{i}") for i in range(num_classes)])
        
        # Set x-axis (subsample for readability)
        step = max(1, len(timesteps) // 20)
        ax.set_xticks(range(0, len(timesteps), step))
        ax.set_xticklabels([timesteps[i] for i in range(0, len(timesteps), step)], rotation=45)
        
        ax.set_xlabel('Timestep (T → 0)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Class', fontsize=12, fontweight='bold')
        ax.set_title('Probability Evolution Heatmap', fontsize=14, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Probability', fontsize=11, fontweight='bold')
        
        # Adjust layout
        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08, wspace=0.3)
        
        # Convert to array
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        
        buf = fig.canvas.buffer_rgba()
        vis_array = np.asarray(buf)[:, :, :3]
        
        plt.close(fig)
        
        if save_path:
            save_visualization(vis_array, save_path, self.dpi)
        
        return vis_array
    
    def create_denoising_animation(self,
                                   explanation: Dict[str, Any],
                                   save_path: Optional[str] = None,
                                   fps: int = 5) -> str:
        """
        Create animated GIF showing the denoising process evolution.
        
        Shows how class probabilities evolve from initial noise to final prediction.
        Each frame shows probabilities at a different timestep.
        
        Args:
            explanation: Explanation dict from DiffusionExplainer
            save_path: Path to save GIF (if None, generates automatically)
            fps: Frames per second for animation
        
        Returns:
            Path to saved GIF file
        """
        trajectory = explanation.get('trajectory', [])
        if len(trajectory) == 0:
            raise ValueError("No trajectory data found in explanation")
        
        timesteps = [step['timestep'] for step in trajectory]
        probs_over_time = np.array([step['probs'] for step in trajectory])
        num_classes = probs_over_time.shape[1]
        
        class_labels = [self.class_names.get(str(i), f"Class {i}") for i in range(num_classes)]
        colors = plt.cm.Set3(np.linspace(0, 1, num_classes))
        
        frames = []
        num_frames = len(trajectory)
        
        print(f"[TrajectoryVisualizer] Creating animation with {num_frames} frames...")
        
        # Use non-interactive backend for better reliability
        import matplotlib
        backend = matplotlib.get_backend()
        matplotlib.use('Agg')  # Non-interactive backend
        
        for frame_idx in range(num_frames):
            # Create fresh figure for each frame to ensure proper updates
            fig, ax = plt.subplots(figsize=(12, 8), dpi=100)  # Lower DPI for faster rendering
            
            probs = probs_over_time[frame_idx]
            timestep = timesteps[frame_idx]
            pred_class = int(np.argmax(probs))
            pred_conf = float(probs[pred_class])
            
            # Draw bars with different colors for predicted class
            bar_colors = [colors[i] if i != pred_class else 'lightcoral' for i in range(num_classes)]
            bars = ax.bar(range(num_classes), probs, color=bar_colors, edgecolor='black', linewidth=1)
            
            # Highlight predicted class with thicker border
            bars[pred_class].set_edgecolor('red')
            bars[pred_class].set_linewidth(3)
            
            # Set labels and formatting
            ax.set_xticks(range(num_classes))
            ax.set_xticklabels(class_labels, rotation=45, ha='right')
            ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
            ax.set_title(f'Denoising Process - Timestep: {timestep}', fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)
            
            # Add text annotations with prediction change indicator
            prediction_changed = (frame_idx > 0 and 
                                 pred_class != int(np.argmax(probs_over_time[frame_idx-1])))
            
            info_text = f'Progress: {frame_idx+1}/{num_frames}\n'
            info_text += f'Predicted: {self.class_names.get(str(pred_class), f"Class {pred_class}")}\n'
            info_text += f'Confidence: {pred_conf:.3f}\n'
            if prediction_changed:
                info_text += 'Status: CHANGED!'
            
            box_color = 'lightcoral' if prediction_changed else 'wheat'
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.7,
                            edgecolor='red' if prediction_changed else 'black',
                            linewidth=2 if prediction_changed else 1))
            
            # Add value labels on top of bars
            for i, (bar, prob) in enumerate(zip(bars, probs)):
                if prob > 0.05:  # Only show if significant
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{prob:.2f}', ha='center', va='bottom', fontsize=9)
            
            # Adjust layout - use subplots_adjust to avoid tight_layout warnings
            fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
            
            # Capture frame - ensure figure is fully rendered
            fig.canvas.draw()
            
            # Get the rendered image from buffer
            buf = fig.canvas.buffer_rgba()
            frame_rgba = np.asarray(buf)
            
            # Convert RGBA to RGB
            frame_rgb = frame_rgba[:, :, :3].copy()
            frames.append(frame_rgb)
            
            plt.close(fig)  # Close immediately to free memory
        
        if save_path is None:
            save_path = 'denoising_animation.gif'
        
        # Verify we have frames
        if len(frames) == 0:
            raise ValueError("No frames generated for animation")
        
        print(f"[TrajectoryVisualizer] Collected {len(frames)} frames, saving GIF...")
        
        # Convert frames to PIL Images for better GIF compatibility
        pil_frames = []
        for i, frame in enumerate(frames):
            # Ensure frame is uint8 and in correct shape
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                else:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
            
            # Ensure frame is (H, W, 3)
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"Frame {i} has invalid shape: {frame.shape}")
            
            pil_frames.append(Image.fromarray(frame, mode='RGB'))
        
        # Save using PIL's save method - this is more reliable for animated GIFs
        duration_ms = int(1000 / fps)  # Duration in milliseconds per frame
        pil_frames[0].save(
            save_path,
            save_all=True,
            append_images=pil_frames[1:] if len(pil_frames) > 1 else [],
            duration=duration_ms,
            loop=0,
            format='GIF'
        )
        
        print(f"[TrajectoryVisualizer] Animation saved to {save_path} with {len(frames)} frames at {fps} fps ({duration_ms}ms per frame)")
        return save_path


"""
Usage Example:

from xai.visualizers.trajectory_vis import TrajectoryVisualizer
from xai.explainers.diffusion_explainer import DiffusionExplainer

# Create visualizer
config = {
    'figure_size': [15, 10],
    'image_dpi': 300,
    'class_names': {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}
}
visualizer = TrajectoryVisualizer(config)

# Generate explanation
explainer = DiffusionExplainer(model, device, config)
explanation = explainer.explain(image_tensor, label=2)

# Create visualizations
vis1 = visualizer.visualize_trajectory(
    explanation,
    save_path='outputs/images/trajectory.png'
)

vis2 = visualizer.visualize_probability_heatmap(
    explanation,
    save_path='outputs/images/prob_heatmap.png'
)

Research Insights from Trajectory Visualization:

1. Decision Timing:
   - Early convergence (high timesteps) → Model decides based on coarse features
   - Late convergence (low timesteps) → Model needs fine details to decide
   - Multiple changes → Uncertainty, potential for error

2. Confidence Patterns:
   - Monotonic increase → Stable, confident decision
   - Oscillations → Model is uncertain, conflicting evidence
   - Sudden jumps → Critical features discovered at specific timesteps

3. Entropy Behavior:
   - High→Low (smooth) → Normal convergence
   - Low→High→Low → Model revises initial guess
   - Remains high → Uncertain, low confidence prediction

4. Model Comparison:
   - Diffusion agrees with aux → Strong evidence
   - Diffusion corrects aux → Refinement helps
   - Aux correct, diffusion wrong → Over-fitting to noise

Future Visualizations:
- Interactive Plotly dashboards
- GIF/MP4 animations of denoising
- 3D trajectory in probability simplex (for 3-4 classes)
- Multi-sample clustering (similar trajectories)
- Attention evolution synchronized with trajectory
"""

