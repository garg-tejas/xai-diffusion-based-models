"""
Main XAI Pipeline

This is the entry point for the XAI analysis framework.
It coordinates all components to generate comprehensive explanations.

Workflow:
1. Load configuration
2. Initialize model loader
3. Select representative samples
4. For each sample:
   - Run attention explainer
   - Run diffusion explainer
   - Generate visualizations
   - Create report
   - Save outputs
5. Generate summary report

Usage:
    python main.py --config config/xai_config.yaml
    python main.py --checkpoint logs/aptos/version_0/checkpoints/last.ckpt --samples 5
    python main.py --help
"""

import sys
import argparse
from pathlib import Path
import yaml
from easydict import EasyDict
from typing import Dict, Any, List
from tqdm import tqdm
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import XAI components
from xai.core.model_loader import ModelLoader
from xai.core.sample_selector import SampleSelector
from xai.explainers.attention_explainer import AttentionExplainer
from xai.explainers.diffusion_explainer import DiffusionExplainer
from xai.visualizers.attention_vis import AttentionVisualizer
from xai.visualizers.trajectory_vis import TrajectoryVisualizer
from xai.visualizers.report_generator import ReportGenerator
from xai.utils.image_utils import load_image, preprocess_for_model


class XAIPipeline:
    """
    Main XAI analysis pipeline.
    
    This class orchestrates all components of the XAI framework to generate
    comprehensive explanations for a set of images.
    
    Attributes:
        config: Configuration dictionary
        model_loader: ModelLoader instance
        sample_selector: SampleSelector instance
        explainers: Dictionary of explainer instances
        visualizers: Dictionary of visualizer instances
        report_generator: ReportGenerator instance
        
    Usage:
        >>> pipeline = XAIPipeline('config/xai_config.yaml')
        >>> pipeline.run()
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the XAI pipeline.
        
        Args:
            config_path: Path to configuration YAML file
        """
        # Load configuration
        config_path = Path(config_path)
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert class_names integer keys to strings for EasyDict compatibility
        if 'data' in config_dict and 'class_names' in config_dict['data']:
            class_names = config_dict['data']['class_names']
            if isinstance(class_names, dict):
                # Convert integer keys to strings
                config_dict['data']['class_names'] = {str(k): v for k, v in class_names.items()}
        
        # Convert balanced_ratio list if present (EasyDict handles lists fine, but ensure it's a list)
        if 'sampling' in config_dict and 'balanced_ratio' in config_dict['sampling']:
            if isinstance(config_dict['sampling']['balanced_ratio'], list):
                # Keep as list, EasyDict will handle it
                pass
        
        self.config = EasyDict(config_dict)
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info("="*80)
        self.logger.info("XAI PIPELINE INITIALIZATION")
        self.logger.info("="*80)
        self.logger.info(f"Config loaded from: {config_path}")
        
        # Initialize components
        self.model_loader = None
        self.model = None
        self.sample_selector = None
        self.explainers = {}
        self.visualizers = {}
        self.report_generator = None
        
        # Output paths
        self.output_dir = Path(__file__).parent / self.config.output.output_dir
        self.images_dir = self.output_dir / self.config.output.images_dir
        self.html_dir = self.output_dir / self.config.output.html_dir
        self.arrays_dir = self.output_dir / self.config.output.arrays_dir
        
        # Create output directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.html_dir.mkdir(parents=True, exist_ok=True)
        self.arrays_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.INFO if self.config.logging.verbosity >= 1 else logging.WARNING
        
        # Create logger
        self.logger = logging.getLogger('XAI')
        self.logger.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if requested
        if self.config.logging.save_log:
            log_file = Path(__file__).parent / self.config.logging.log_file
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(console_formatter)
            self.logger.addHandler(file_handler)
    
    def initialize_model(self):
        """Load the trained model from checkpoint."""
        self.logger.info("\n" + "-"*80)
        self.logger.info("LOADING MODEL")
        self.logger.info("-"*80)
        
        # Construct paths (relative to project root)
        project_root = Path(__file__).parent.parent
        checkpoint_path = self.config.model.checkpoint_path
        config_path = self.config.model.config_path
        
        # Initialize model loader
        self.model_loader = ModelLoader(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=self.config.model.device
        )
        
        # Load model
        self.model = self.model_loader.load_model()
        
        # Print model info
        info = self.model_loader.get_model_info()
        self.logger.info(f"Model info:")
        for key, value in info.items():
            self.logger.info(f"  {key}: {value}")
    
    def initialize_sample_selector(self):
        """Initialize sample selector and select samples."""
        self.logger.info("\n" + "-"*80)
        self.logger.info("SELECTING SAMPLES")
        self.logger.info("-"*80)
        
        self.sample_selector = SampleSelector(
            csv_path=self.config.sampling.predictions_csv,
            num_classes=self.config.data.num_classes,
            random_seed=self.config.sampling.random_seed
        )
        
        # Select samples
        self.selected_samples = self.sample_selector.select_samples(
            samples_per_class=self.config.sampling.samples_per_class,
            strategy=self.config.sampling.selection_strategy,
            balanced_ratio=self.config.sampling.balanced_ratio
        )
        
        self.logger.info(f"Selected {len(self.selected_samples)} samples for analysis")
    
    def initialize_explainers(self):
        """Initialize explainer modules."""
        self.logger.info("\n" + "-"*80)
        self.logger.info("INITIALIZING EXPLAINERS")
        self.logger.info("-"*80)
        
        device = self.model_loader.device
        
        # Attention explainer
        if self.config.explainers.attention.enabled:
            self.explainers['attention'] = AttentionExplainer(
                model=self.model,
                device=device,
                config={'num_classes': self.config.data.num_classes}
            )
            self.logger.info("[OK] Attention explainer initialized")
        
        # Diffusion explainer
        if self.config.explainers.diffusion.enabled:
            self.explainers['diffusion'] = DiffusionExplainer(
                model=self.model,
                device=device,
                config={'subsample_timesteps': self.config.explainers.diffusion.num_timesteps_to_visualize}
            )
            self.logger.info("[OK] Diffusion explainer initialized")
    
    def initialize_visualizers(self):
        """Initialize visualization modules."""
        self.logger.info("\n" + "-"*80)
        self.logger.info("INITIALIZING VISUALIZERS")
        self.logger.info("-"*80)
        
        vis_config = {
            'colormap': self.config.visualization.colormap,
            'overlay_alpha': self.config.visualization.overlay_alpha,
            'figure_size': self.config.visualization.figure_size,
            'image_dpi': self.config.output.image_dpi,
            'class_names': self.config.data.class_names,
        }
        
        self.visualizers['attention'] = AttentionVisualizer(vis_config)
        self.visualizers['trajectory'] = TrajectoryVisualizer(vis_config)
        self.report_generator = ReportGenerator(vis_config)
        
        self.logger.info("[OK] All visualizers initialized")
    
    def process_sample(self, sample_row, sample_idx: int) -> Dict[str, Any]:
        """
        Process a single sample through the XAI pipeline.
        
        Args:
            sample_row: Row from selected samples DataFrame
            sample_idx: Index of this sample
        
        Returns:
            Dictionary with all results and paths
        """
        # Get sample info
        img_path_rel = sample_row['img_path']
        img_path = Path(__file__).parent.parent / img_path_rel
        ground_truth = int(sample_row['label'])
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Processing sample {sample_idx + 1}/{len(self.selected_samples)}")
        self.logger.info(f"Image: {img_path.name}")
        # Convert ground_truth to string for class_names lookup (keys are strings after EasyDict conversion)
        gt_str = str(ground_truth) if ground_truth != -1 else -1
        gt_name = self.config.data.class_names.get(gt_str, f"Class {ground_truth}") if gt_str != -1 else "Unknown"
        self.logger.info(f"Ground truth: {gt_name}")
        self.logger.info(f"{'='*60}")
        
        # Load and preprocess image
        image_pil = load_image(
            img_path,
            target_size=(self.config.data.image_size, self.config.data.image_size)
        )
        image_tensor = preprocess_for_model(
            image_pil,
            normalize_mean=self.config.data.normalize_mean,
            normalize_std=self.config.data.normalize_std
        )
        
        # Move to device
        image_tensor = image_tensor.to(self.model_loader.device)
        
        results = {
            'img_path': str(img_path_rel),
            'ground_truth': ground_truth,
            'sample_idx': sample_idx
        }
        
        visualizations = {}
        
        # Run attention explainer
        if 'attention' in self.explainers:
            self.logger.info("Running attention explainer...")
            attention_exp = self.explainers['attention'].explain(image_tensor, label=ground_truth)
            results['attention_exp'] = attention_exp
            results['attention_prediction'] = attention_exp['prediction']
            results['attention_confidence'] = attention_exp['confidence']
            results['attention_correct'] = (attention_exp['prediction'] == ground_truth)
            
            # Generate visualizations
            import numpy as np
            image_array = np.array(image_pil)
            vis = self.visualizers['attention'].visualize_attention(
                image_array, attention_exp,
                save_path=self.images_dir / f"class_{ground_truth}_sample_{sample_idx}_attention.png"
            )
            visualizations['attention_overview'] = vis
            
            # Create interactive visualization if enabled
            if self.config.output.create_interactive:
                try:
                    interactive_path = self.html_dir / f"class_{ground_truth}_sample_{sample_idx}_interactive.html"
                    self.visualizers['attention'].create_interactive_visualization(
                        image_array, attention_exp,
                        save_path=interactive_path
                    )
                    results['interactive_attention_path'] = str(interactive_path.relative_to(self.output_dir))
                except Exception as e:
                    self.logger.warning(f"Failed to create interactive visualization: {e}")
        
        # Run diffusion explainer
        if 'diffusion' in self.explainers:
            self.logger.info("Running diffusion explainer...")
            diffusion_exp = self.explainers['diffusion'].explain(image_tensor, label=ground_truth)
            results['diffusion_exp'] = diffusion_exp
            results['diffusion_prediction'] = diffusion_exp['prediction']
            results['diffusion_confidence'] = diffusion_exp['confidence']
            results['diffusion_correct'] = (diffusion_exp['prediction'] == ground_truth)
            
            # Generate visualizations
            vis = self.visualizers['trajectory'].visualize_trajectory(
                diffusion_exp,
                save_path=self.images_dir / f"class_{ground_truth}_sample_{sample_idx}_trajectory.png"
            )
            visualizations['trajectory'] = vis
            
            # Create denoising animation if enabled
            if self.config.output.create_animations:
                try:
                    animation_path = self.images_dir / f"class_{ground_truth}_sample_{sample_idx}_denoising.gif"
                    self.visualizers['trajectory'].create_denoising_animation(
                        diffusion_exp,
                        save_path=animation_path,
                        fps=self.config.output.animation_fps
                    )
                    results['denoising_animation_path'] = str(animation_path.relative_to(self.output_dir))
                except Exception as e:
                    self.logger.warning(f"Failed to create denoising animation: {e}")
            
            # Create attention evolution animation if diffusion explainer tracks attention
            if self.config.output.create_animations and 'attention_evolution' in diffusion_exp:
                try:
                    import numpy as np
                    image_array = np.array(image_pil)
                    attn_evolution_path = self.images_dir / f"class_{ground_truth}_sample_{sample_idx}_attention_evolution.gif"
                    self.visualizers['attention'].create_attention_evolution_animation(
                        image_array,
                        diffusion_exp['attention_evolution'],
                        save_path=attn_evolution_path,
                        fps=self.config.output.animation_fps
                    )
                    results['attention_evolution_path'] = str(attn_evolution_path.relative_to(self.output_dir))
                except Exception as e:
                    self.logger.warning(f"Failed to create attention evolution animation: {e}")
        
        # Generate HTML report
        if self.config.output.save_html:
            self.logger.info("Generating HTML report...")
            report_path = self.html_dir / f"class_{ground_truth}_sample_{sample_idx}_report.html"
            
            sample_info = {
                'img_path': img_path_rel,
                'ground_truth': ground_truth
            }
            
            self.report_generator.create_sample_report(
                sample_info=sample_info,
                attention_exp=results.get('attention_exp', {}),
                diffusion_exp=results.get('diffusion_exp', {}),
                visualizations=visualizations,
                save_path=report_path
            )
            results['report_path'] = str(report_path.relative_to(self.output_dir))
        
        # Save arrays if requested
        if self.config.output.save_arrays:
            import numpy as np
            array_path = self.arrays_dir / f"class_{ground_truth}_sample_{sample_idx}_data.npz"
            np.savez(
                array_path,
                attention_saliency=results.get('attention_exp', {}).get('saliency_map'),
                diffusion_trajectory=[step['probs'] for step in results.get('diffusion_exp', {}).get('trajectory', [])]
            )
        
        self.logger.info(f"[OK] Sample {sample_idx + 1} complete")
        
        return results
    
    def generate_summary_report(self, all_results: List[Dict[str, Any]]):
        """Generate summary HTML report."""
        self.logger.info("\n" + "="*80)
        self.logger.info("GENERATING SUMMARY REPORT")
        self.logger.info("="*80)
        
        summary_path = self.html_dir / "index.html"
        self.report_generator.create_summary_report(all_results, summary_path)
        
        self.logger.info(f"[OK] Summary report saved to {summary_path}")
    
    def print_summary_statistics(self, all_results: List[Dict[str, Any]]):
        """Print summary statistics."""
        self.logger.info("\n" + "="*80)
        self.logger.info("SUMMARY STATISTICS")
        self.logger.info("="*80)
        
        total = len(all_results)
        
        # Accuracy
        att_correct = sum(1 for r in all_results if r.get('attention_correct', False))
        diff_correct = sum(1 for r in all_results if r.get('diffusion_correct', False))
        
        self.logger.info(f"Total samples analyzed: {total}")
        self.logger.info(f"Auxiliary model accuracy: {att_correct}/{total} ({att_correct/total*100:.1f}%)")
        self.logger.info(f"Diffusion model accuracy: {diff_correct}/{total} ({diff_correct/total*100:.1f}%)")
        
        # Average confidence
        import numpy as np
        avg_att_conf = np.mean([r.get('attention_confidence', 0) for r in all_results])
        avg_diff_conf = np.mean([r.get('diffusion_confidence', 0) for r in all_results])
        
        self.logger.info(f"Average auxiliary confidence: {avg_att_conf:.3f}")
        self.logger.info(f"Average diffusion confidence: {avg_diff_conf:.3f}")
        
        # Agreement
        agreement = sum(1 for r in all_results 
                       if r.get('attention_prediction') == r.get('diffusion_prediction'))
        self.logger.info(f"Model agreement: {agreement}/{total} ({agreement/total*100:.1f}%)")
    
    def run(self):
        """Run the complete XAI pipeline."""
        start_time = datetime.now()
        
        try:
            # Initialize all components
            self.initialize_model()
            self.initialize_sample_selector()
            self.initialize_explainers()
            self.initialize_visualizers()
            
            # Process all samples
            self.logger.info("\n" + "="*80)
            self.logger.info("PROCESSING SAMPLES")
            self.logger.info("="*80)
            
            all_results = []
            
            if self.config.logging.show_progress:
                iterator = tqdm(self.selected_samples.iterrows(), 
                              total=len(self.selected_samples),
                              desc="Processing samples")
            else:
                iterator = self.selected_samples.iterrows()
            
            for idx, row in iterator:
                results = self.process_sample(row, idx)
                all_results.append(results)
            
            # Generate summary report
            if self.config.output.save_html:
                self.generate_summary_report(all_results)
            
            # Print statistics
            self.print_summary_statistics(all_results)
            
            # Save configuration used
            config_save_path = self.output_dir / "run_config.yaml"
            with open(config_save_path, 'w') as f:
                yaml.dump(dict(self.config), f, default_flow_style=False)
            self.logger.info(f"\n[OK] Configuration saved to {config_save_path}")
            
        except Exception as e:
            self.logger.error(f"\n[ERROR] Pipeline failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.info("\n" + "="*80)
            self.logger.info(f"PIPELINE COMPLETE")
            self.logger.info(f"Duration: {duration:.1f} seconds")
            self.logger.info(f"Output directory: {self.output_dir}")
            self.logger.info("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='XAI Analysis Pipeline for Diffusion-Based Classification')
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/xai_config.yaml',
        help='Path to configuration file (default: config/xai_config.yaml)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Override checkpoint path from config'
    )
    parser.add_argument(
        '--samples',
        type=int,
        help='Override number of samples per class'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory'
    )
    
    args = parser.parse_args()
    
    # Construct config path (relative to xai directory)
    config_path = Path(__file__).parent / args.config
    
    # Create pipeline
    pipeline = XAIPipeline(config_path)
    
    # Apply command-line overrides
    if args.checkpoint:
        pipeline.config.model.checkpoint_path = args.checkpoint
    if args.samples:
        pipeline.config.sampling.samples_per_class = args.samples
    if args.output_dir:
        pipeline.config.output.output_dir = args.output_dir
    
    # Run pipeline
    pipeline.run()


if __name__ == '__main__':
    main()

