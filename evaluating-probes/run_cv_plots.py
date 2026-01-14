#!/usr/bin/env python3
"""
Cross-validation plot runner for specific datasets and comparisons.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from src.visualize.hyperparameter_analysis import CrossValidationAnalyzer

def main():
    """Run cross-validation plots for specific datasets."""
    print("Running cross-validation plots for 94_better_spam and 98_mask_all_honesty...")
    
    # Initialize the analyzer
    analyzer = CrossValidationAnalyzer()
    
    # Focus only on the specific datasets and models
    target_datasets = ['94_better_spam', '98_mask_all_honesty']
    target_models = ['spam_gemma_9b', 'mask_llama33_70b']
    
    # Run cross-validation plots for each dataset-model combination
    for dataset in target_datasets:
        for model in target_models:
            print(f"\nProcessing {dataset} with {model}...")
            
            try:
                # Generate cross-validation plots for both AUC and Recall
                for metric in ['auc', 'recall']:
                    print(f"  Generating {metric.upper()} joint comparison plot...")
                    
                    # Create output directory
                    output_dir = Path("visualizations/hyp")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Generate joint comparison plots (experiment 2 vs 4)
                    save_path = output_dir / f"cv_{dataset}_{model}_comparison_{metric}.png"
                    analyzer.plot_experiment_comparison_with_cv_hyperparameters(
                        eval_dataset=dataset,
                        run_name=model,
                        save_path=save_path,
                        metric=metric
                    )
                    
            except Exception as e:
                print(f"Error processing {dataset} with {model}: {e}")
    
    print("\nCross-validation plot generation complete!")

if __name__ == "__main__":
    main()
