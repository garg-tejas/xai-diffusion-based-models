"""
Noise Visualization Module

Creates visualizations for heterologous noise patterns and noise reduction
during the diffusion denoising process.

Features:
- Noise interaction maps showing spatial correlations
- Timestep distribution heatmaps
- Noise magnitude evolution plots
- Spatial noise pattern visualizations
- Noise reduction animations through diffusion timesteps
- Multi-panel noise analysis views

Purpose:
- Visualize how heterologous noise affects diffusion
- Understand noise reduction process
- Analyze spatial noise patterns and interactions
- Support research on diffusion dynamics
- Enable noise pattern debugging and analysis

Future Extensions:
- 3D noise visualization
- Interactive noise exploration
- Noise pattern clustering visualization
- Multi-sample noise comparison
- Noise-guided feature overlays
- Export to video formats (MP4, WebM)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, Any, Optional, List
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from xai.utils.image_utils import figure_to_array
from xai.utils.animation_utils import capture_frame, save_frames_as_gif


class NoiseVisualizer:
    """
    Visualize heterologous noise patterns and noise reduction.
    
    Creates visualizations showing:
    - Noise interaction maps
    - Timestep distributions
    - Noise magnitude evolution
    - Spatial noise patterns
    - Noise reduction animations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the noise visualizer.
        
        Args:
            config: Configuration dictionary with visualization settings
        """
        self.config = config
        self.colormap = config.get('colormap', 'viridis')
        self.figsize = config.get('figure_size', [12, 8])
        self.class_names = config.get('class_names', {})
        
        if not self.class_names:
            self.class_names = {str(i): f"Class {i}" for i in range(10)}
    
    def create_noise_interaction_map(self,
                                    noise_map: np.ndarray,
                                    save_path: Optional[str] = None) -> np.ndarray:
        """
        Show how noise points interact via spatial correlation.
        
        Args:
            noise_map: Noise map [C, H, W] or [H, W]
            save_path: Optional path to save visualization
            
        Returns:
            Visualization as numpy array
        """
        if noise_map.ndim == 3:
            # Average over classes or show for first class
            noise_map = noise_map[0]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Noise map
        im1 = ax1.imshow(noise_map, cmap='hot', aspect='auto')
        ax1.set_title('Noise Magnitude Map', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Width', fontsize=12)
        ax1.set_ylabel('Height', fontsize=12)
        plt.colorbar(im1, ax=ax1, label='Noise Magnitude')
        
        # Plot 2: Correlation with neighbors
        h, w = noise_map.shape
        flat_noise = noise_map.flatten()
        
        # Compute correlation with shifted versions
        shifts = [1, w, w+1, w-1]  # Right, down, diagonal
        correlations = []
        for shift in shifts:
            shifted = np.roll(flat_noise, shift)
            if shift > 0:
                shifted[:shift] = 0
            elif shift < 0:
                shifted[shift:] = 0
            corr = np.corrcoef(flat_noise, shifted)[0, 1]
            correlations.append(corr)
        
        # Create correlation visualization
        directions = ['Right', 'Down', 'Diag+', 'Diag-']
        bars = ax2.bar(directions, correlations, color='steelblue', alpha=0.8, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Correlation', fontsize=12)
        ax2.set_title('Spatial Noise Correlation', fontsize=14, fontweight='bold')
        ax2.set_ylim(-1, 1)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, correlations):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05 if height >= 0 else height - 0.05,
                    f'{val:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10, fontweight='bold')
        
        fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.1, wspace=0.3)
        vis_array = figure_to_array(fig)
        plt.close(fig)
        
        if save_path:
            Image.fromarray(vis_array).save(save_path)
        
        return vis_array
    
    def create_timestep_distribution(self,
                                    timestep_map: np.ndarray,
                                    save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize random timestep sampling across spatial locations.
        
        Args:
            timestep_map: Timestep values at each location [H, W]
            save_path: Optional path to save visualization
            
        Returns:
            Visualization as numpy array
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(timestep_map, cmap='plasma', aspect='auto')
        ax.set_title('Timestep Distribution Across Spatial Locations', fontsize=14, fontweight='bold')
        ax.set_xlabel('Width', fontsize=12)
        ax.set_ylabel('Height', fontsize=12)
        plt.colorbar(im, ax=ax, label='Timestep')
        
        fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.1)
        vis_array = figure_to_array(fig)
        plt.close(fig)
        
        if save_path:
            Image.fromarray(vis_array).save(save_path)
        
        return vis_array
    
    def create_noise_magnitude_plot(self,
                                   noise_magnitudes: List[float],
                                   timesteps: List[int],
                                   save_path: Optional[str] = None) -> np.ndarray:
        """
        Show noise magnitude evolution during denoising.
        
        Args:
            noise_magnitudes: List of noise magnitudes at each timestep
            timesteps: List of timestep values
            save_path: Optional path to save visualization
            
        Returns:
            Visualization as numpy array
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Reverse timesteps for display (from high to low)
        if len(timesteps) > 1 and timesteps[0] > timesteps[-1]:
            # Already in descending order
            x_vals = timesteps
        else:
            # Reverse if needed
            x_vals = timesteps[::-1]
            noise_magnitudes = noise_magnitudes[::-1]
        
        ax.plot(x_vals, noise_magnitudes, marker='o', linewidth=2, markersize=4, color='coral')
        ax.set_xlabel('Timestep', fontsize=12)
        ax.set_ylabel('Noise Magnitude', fontsize=12)
        ax.set_title('Noise Reduction Through Denoising', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_xlim(x_vals[0], x_vals[-1])
        
        # Add trend line
        if len(noise_magnitudes) > 1:
            z = np.polyfit(x_vals, noise_magnitudes, 1)
            p = np.poly1d(z)
            ax.plot(x_vals, p(x_vals), "r--", alpha=0.5, label=f'Trend (slope: {z[0]:.4f})')
            ax.legend(fontsize=10)
        
        fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.1)
        vis_array = figure_to_array(fig)
        plt.close(fig)
        
        if save_path:
            Image.fromarray(vis_array).save(save_path)
        
        return vis_array
    
    def create_spatial_noise_animation(self,
                                      noise_data: List[Dict[str, Any]],
                                      save_path: Optional[str] = None,
                                      fps: int = 1) -> str:
        """
        Animate noise reduction across 2D space through diffusion timesteps.
        
        Args:
            noise_data: List of noise data dictionaries from NoiseExplainer
            save_path: Path to save GIF
            fps: Frames per second (default 1 for slow viewing)
            
        Returns:
            Path to saved GIF
        """
        import matplotlib
        matplotlib.use('Agg')
        
        if len(noise_data) == 0:
            raise ValueError("No noise data provided for animation")
        
        frames = []
        num_frames = len(noise_data)
        
        for frame_idx, step in enumerate(noise_data):
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12), dpi=100)
            
            timestep = step.get('timestep', frame_idx)
            noise_map = step.get('noise_map', None)
            noise_magnitude = step.get('noise_magnitude', None)
            timestep_map = step.get('timestep_map', None)
            probs = step.get('probs', None)
            pred_class = step.get('predicted_class', 0)
            
            # Top left: Noise magnitude map
            if noise_magnitude is not None:
                im1 = ax1.imshow(noise_magnitude, cmap='hot', aspect='auto')
                ax1.set_title('Noise Magnitude', fontsize=11, fontweight='bold')
                ax1.set_xlabel('Width', fontsize=9)
                ax1.set_ylabel('Height', fontsize=9)
                plt.colorbar(im1, ax=ax1, fraction=0.046)
            
            # Top right: Timestep map
            if timestep_map is not None:
                im2 = ax2.imshow(timestep_map, cmap='plasma', aspect='auto')
                ax2.set_title('Timestep Distribution', fontsize=11, fontweight='bold')
                ax2.set_xlabel('Width', fontsize=9)
                ax2.set_ylabel('Height', fontsize=9)
                plt.colorbar(im2, ax=ax2, fraction=0.046)
            
            # Bottom left: Noise map (per class if available)
            if noise_map is not None:
                if noise_map.ndim == 3:
                    # Show for predicted class
                    noise_class = noise_map[pred_class]
                else:
                    noise_class = noise_map
                
                im3 = ax3.imshow(noise_class, cmap='viridis', aspect='auto')
                ax3.set_title(f'Noise Map (Class {pred_class})', fontsize=11, fontweight='bold')
                ax3.set_xlabel('Width', fontsize=9)
                ax3.set_ylabel('Height', fontsize=9)
                plt.colorbar(im3, ax=ax3, fraction=0.046)
            
            # Bottom right: Class probabilities
            if probs is not None:
                num_classes = len(probs)
                colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
                bar_colors = [colors[i] if i != pred_class else 'lightcoral' for i in range(num_classes)]
                
                bars = ax4.bar(range(num_classes), probs, color=bar_colors, edgecolor='black', linewidth=1)
                bars[pred_class].set_edgecolor('red')
                bars[pred_class].set_linewidth(3)
                
                ax4.set_xticks(range(num_classes))
                ax4.set_xticklabels([self.class_names.get(str(i), f"C{i}") for i in range(num_classes)],
                                  rotation=45, ha='right', fontsize=8)
                ax4.set_ylabel('Probability', fontsize=10)
                ax4.set_title('Class Probabilities', fontsize=11, fontweight='bold')
                ax4.set_ylim(0, 1)
                ax4.grid(axis='y', alpha=0.3)
            
            fig.suptitle(f'Noise Evolution - Timestep: {timestep} | Step: {frame_idx+1}/{num_frames}',
                        fontsize=14, fontweight='bold')
            fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.08, hspace=0.3, wspace=0.3)
            
            frame = capture_frame(fig)
            frames.append(frame)
            plt.close(fig)
            
            if (frame_idx + 1) % max(1, num_frames // 10) == 0:
                print(f"  Frame {frame_idx + 1}/{num_frames} captured")
        
        if save_path is None:
            save_path = 'noise_evolution.gif'
        
        print(f"[NoiseVisualizer] Saving animation to {save_path}...")
        save_frames_as_gif(frames, save_path, fps=fps)
        
        return save_path

