"""
Feature Prior Visualization Module

Creates visualizations for feature prior composition showing how
Transformer and CNN features are fused.

Features:
- Feature contribution plots (raw vs ROI contributions)
- Fusion weight heatmaps showing learnable parameters
- Feature space comparisons using PCA/t-SNE
- ROI contribution rankings and importance bars
- Feature magnitude comparisons
- Animation of feature evolution through diffusion

Purpose:
- Visualize feature fusion mechanisms
- Understand contribution of different feature sources
- Identify important ROI patches
- Support feature engineering decisions
- Enable research on multi-modal feature learning
- Facilitate model debugging and optimization

Future Extensions:
- Interactive feature space exploration
- 3D feature space visualization
- Feature trajectory animations
- Multi-sample feature comparison
- Feature attention overlays on images
- Export to interactive HTML with Plotly
- Feature correlation matrices
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, Any, Optional, List
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from xai.utils.image_utils import figure_to_array
from xai.utils.animation_utils import capture_frame, save_frames_as_gif


class FeaturePriorVisualizer:
    """
    Visualize feature prior composition and fusion.
    
    Creates visualizations showing:
    - Feature contribution plots
    - Fusion weight heatmaps
    - Feature space comparisons (t-SNE/PCA)
    - ROI contribution rankings
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature prior visualizer.
        
        Args:
            config: Configuration dictionary with visualization settings
        """
        self.config = config
        self.colormap = config.get('colormap', 'viridis')
        self.figsize = config.get('figure_size', [12, 8])
        self.class_names = config.get('class_names', {})
        
        if not self.class_names:
            self.class_names = {str(i): f"Class {i}" for i in range(10)}
    
    def create_feature_contribution_plot(self,
                                        contribution_scores: Dict[str, float],
                                        save_path: Optional[str] = None) -> np.ndarray:
        """
        Bar chart showing raw vs ROI feature contributions.
        
        Args:
            contribution_scores: Dictionary with contribution scores
            save_path: Optional path to save visualization
            
        Returns:
            Visualization as numpy array
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Contribution percentages
        categories = ['Raw Features\n(Transformer)', 'ROI Features\n(CNN)']
        contributions = [
            contribution_scores.get('raw_contribution', 0.5),
            contribution_scores.get('roi_contribution', 0.5)
        ]
        colors = ['skyblue', 'lightcoral']
        
        bars = ax1.bar(categories, contributions, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax1.set_ylabel('Contribution (%)', fontsize=12)
        ax1.set_title('Feature Contribution to Fusion', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, contributions):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Plot 2: Magnitudes and weights
        metrics = ['Magnitude', 'Weight']
        raw_values = [
            contribution_scores.get('raw_magnitude', 0),
            contribution_scores.get('raw_weight', 0)
        ]
        roi_values = [
            contribution_scores.get('roi_magnitude', 0),
            contribution_scores.get('roi_weight', 0)
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, raw_values, width, label='Raw', color='skyblue', alpha=0.8)
        bars2 = ax2.bar(x + width/2, roi_values, width, label='ROI', color='lightcoral', alpha=0.8)
        
        ax2.set_ylabel('Value', fontsize=12)
        ax2.set_title('Feature Magnitudes and Weights', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend(fontsize=11)
        ax2.grid(axis='y', alpha=0.3)
        
        # Normalize for display if needed
        max_val = max(max(raw_values), max(roi_values))
        if max_val > 0:
            ax2.set_ylim(0, max_val * 1.1)
        
        fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.1, wspace=0.3)
        vis_array = figure_to_array(fig)
        plt.close(fig)
        
        if save_path:
            Image.fromarray(vis_array).save(save_path)
        
        return vis_array
    
    def create_fusion_weight_heatmap(self,
                                    fusion_weights: np.ndarray,
                                    save_path: Optional[str] = None) -> np.ndarray:
        """
        Heatmap of learnable fusion weights Q.
        
        Args:
            fusion_weights: Fusion weights [feature_dim, K+1] or [1, feature_dim, K+1]
            save_path: Optional path to save visualization
            
        Returns:
            Visualization as numpy array
        """
        # Handle different input shapes
        if fusion_weights.ndim == 3:
            fusion_weights = fusion_weights.squeeze(0)  # [feature_dim, K+1]
        
        feature_dim, num_sources = fusion_weights.shape
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(fusion_weights, cmap=self.colormap, aspect='auto')
        ax.set_title('Fusion Weight Matrix Q', fontsize=14, fontweight='bold')
        ax.set_xlabel('Feature Source (0=Raw, 1-6=ROIs)', fontsize=12)
        ax.set_ylabel('Feature Dimension', fontsize=12)
        plt.colorbar(im, ax=ax, label='Weight')
        
        # Add labels
        ax.set_xticks(range(num_sources))
        ax.set_xticklabels(['Raw'] + [f'ROI{i}' for i in range(1, num_sources)])
        
        # Subsample y-axis labels if too many
        if feature_dim > 20:
            step = feature_dim // 10
            ax.set_yticks(range(0, feature_dim, step))
        
        fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.15)
        vis_array = figure_to_array(fig)
        plt.close(fig)
        
        if save_path:
            Image.fromarray(vis_array).save(save_path)
        
        return vis_array
    
    def create_feature_comparison(self,
                                 raw_features: np.ndarray,
                                 roi_features: np.ndarray,
                                 save_path: Optional[str] = None,
                                 method: str = 'pca') -> np.ndarray:
        """
        t-SNE/PCA comparing Transformer vs CNN features.
        
        Args:
            raw_features: Raw features [feature_dim, H, W] or flattened
            roi_features: ROI features [num_patches, feature_dim]
            save_path: Optional path to save visualization
            method: 'pca' or 'tsne'
            
        Returns:
            Visualization as numpy array
        """
        # Flatten and prepare features
        if raw_features.ndim > 2:
            # Average pool to get global representation
            raw_flat = raw_features.reshape(raw_features.shape[0], -1).mean(axis=1)
        else:
            raw_flat = raw_features.flatten()
        
        # Prepare data for dimensionality reduction
        # Combine raw and ROI features
        num_rois = roi_features.shape[0] if roi_features.ndim > 1 else 1
        
        # If ROI features are 2D, use them directly
        if roi_features.ndim == 2:
            roi_flat = roi_features  # [num_patches, feature_dim]
        else:
            roi_flat = roi_features.reshape(1, -1)
        
        # Ensure same feature dimension
        min_dim = min(len(raw_flat), roi_flat.shape[1] if roi_flat.ndim > 1 else len(roi_flat))
        raw_flat = raw_flat[:min_dim]
        
        if roi_flat.ndim == 1:
            roi_flat = roi_flat[:min_dim].reshape(1, -1)
        else:
            roi_flat = roi_flat[:, :min_dim]
        
        # Combine for visualization
        all_features = np.vstack([
            raw_flat.reshape(1, -1),
            roi_flat
        ])
        
        # Reduce dimensionality
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            reduced = reducer.fit_transform(all_features)
            explained_var = reducer.explained_variance_ratio_.sum()
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features)-1))
            reduced = reducer.fit_transform(all_features)
            explained_var = None
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot raw features
        ax.scatter(reduced[0, 0], reduced[0, 1], 
                  c='blue', s=200, marker='*', label='Raw (Transformer)', 
                  edgecolors='black', linewidths=2, zorder=3)
        
        # Plot ROI features
        if len(reduced) > 1:
            ax.scatter(reduced[1:, 0], reduced[1:, 1],
                      c='red', s=100, marker='o', label='ROIs (CNN)',
                      edgecolors='black', linewidths=1, alpha=0.7, zorder=2)
        
        ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
        ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
        title = f'Feature Space Comparison ({method.upper()})'
        if explained_var is not None:
            title += f' (Explained Variance: {explained_var:.2%})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.1)
        vis_array = figure_to_array(fig)
        plt.close(fig)
        
        if save_path:
            Image.fromarray(vis_array).save(save_path)
        
        return vis_array
    
    def create_roi_contribution_bars(self,
                                    roi_features: np.ndarray,
                                    patch_attention: Optional[np.ndarray] = None,
                                    save_path: Optional[str] = None) -> np.ndarray:
        """
        Show which ROIs contribute most to final prediction.
        
        Args:
            roi_features: ROI features [num_patches, feature_dim]
            patch_attention: Patch attention weights [num_patches] (optional)
            save_path: Optional path to save visualization
            
        Returns:
            Visualization as numpy array
        """
        num_patches = roi_features.shape[0] if roi_features.ndim > 1 else 1
        
        # Compute feature magnitudes for each ROI
        if roi_features.ndim > 1:
            roi_magnitudes = np.linalg.norm(roi_features, axis=1)
        else:
            roi_magnitudes = np.array([np.linalg.norm(roi_features)])
        
        # Normalize
        if roi_magnitudes.max() > 0:
            roi_magnitudes = roi_magnitudes / roi_magnitudes.max()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Feature magnitudes
        x = np.arange(num_patches)
        bars1 = ax1.bar(x, roi_magnitudes, color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('ROI Index', fontsize=12)
        ax1.set_ylabel('Normalized Feature Magnitude', fontsize=12)
        ax1.set_title('ROI Feature Magnitudes', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'ROI {i+1}' for i in range(num_patches)])
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, val in zip(bars1, roi_magnitudes):
            if val > 0.05:
                ax1.text(bar.get_x() + bar.get_width()/2., val + 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Attention weights (if available)
        if patch_attention is not None:
            if patch_attention.ndim > 1:
                patch_attention = patch_attention.flatten()
            patch_attention = patch_attention[:num_patches]
            
            # Normalize
            if patch_attention.max() > 0:
                patch_attention = patch_attention / patch_attention.max()
            
            bars2 = ax2.bar(x, patch_attention, color='coral', alpha=0.8, edgecolor='black', linewidth=1.5)
            ax2.set_xlabel('ROI Index', fontsize=12)
            ax2.set_ylabel('Normalized Attention Weight', fontsize=12)
            ax2.set_title('ROI Attention Weights', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels([f'ROI {i+1}' for i in range(num_patches)])
            ax2.grid(axis='y', alpha=0.3)
            ax2.set_ylim(0, 1.1)
            
            # Add value labels
            for bar, val in zip(bars2, patch_attention):
                if val > 0.05:
                    ax2.text(bar.get_x() + bar.get_width()/2., val + 0.02,
                            f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'Attention weights\nnot available', 
                    ha='center', va='center', fontsize=14, transform=ax2.transAxes)
            ax2.set_title('ROI Attention Weights', fontsize=14, fontweight='bold')
        
        fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.1, wspace=0.3)
        vis_array = figure_to_array(fig)
        plt.close(fig)
        
        if save_path:
            Image.fromarray(vis_array).save(save_path)
        
        return vis_array

