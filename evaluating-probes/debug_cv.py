#!/usr/bin/env python3
"""
Debug script to investigate cross-validation hyperparameter selection.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from src.visualize.hyperparameter_analysis import CrossValidationAnalyzer
from src.visualize.data_loader import get_data_for_visualization
from src.visualize.viz_util import apply_main_plot_filters

def debug_attention_cv():
    """Debug attention probe cross-validation selection."""
    print("=== DEBUGGING ATTENTION PROBE CROSS-VALIDATION ===\n")
    
    # Test with a specific case
    dataset = '94_better_spam'
    model = 'spam_gemma_9b'
    experiment = '2-'
    
    analyzer = CrossValidationAnalyzer()
    
    # Get val_eval data for cross-validation
    val_df = get_data_for_visualization(
        eval_dataset=dataset,
        experiment=experiment,
        run_name=model,
        exclude_attention=False,
        include_val_eval=True
    )
    
    # Filter to val_eval results
    val_df = val_df[val_df['filename'].str.contains('/val_eval/')]
    
    print(f"Total val_eval rows: {len(val_df)}")
    
    # Filter to attention probes
    attention_df = val_df[val_df['probe_name'].str.contains('attention', na=False)]
    print(f"Attention probe rows: {len(attention_df)}")
    
    if not attention_df.empty:
        print(f"Available lr values: {sorted(attention_df['lr'].dropna().unique())}")
        print(f"Available weight_decay values: {sorted(attention_df['weight_decay'].dropna().unique())}")
        print(f"Available sample counts: {sorted(attention_df['num_positive_samples'].dropna().unique())}")
        
        # Show performance by lr/weight_decay combination
        lr_wd_performance = attention_df.groupby(['lr', 'weight_decay'])['auc'].agg(['mean', 'std', 'count'])
        print("\nPerformance by lr/weight_decay combination:")
        print(lr_wd_performance.sort_values('mean', ascending=False))
        
        # Show performance by sample count for each lr/wd combination
        print("\nPerformance by sample count for each lr/wd combination:")
        for (lr, wd), group in attention_df.groupby(['lr', 'weight_decay']):
            print(f"\nlr={lr:.1e}, wd={wd:.1e}:")
            sample_perf = group.groupby('num_positive_samples')['auc'].agg(['mean', 'std', 'count'])
            print(sample_perf)
    
    # Now get the CV-selected hyperparameters
    print("\n=== CROSS-VALIDATION SELECTION ===")
    best_hyperparams = analyzer.get_cross_validation_best_hyperparameters(
        dataset, experiment, model
    )
    print(f"Selected hyperparameters: {best_hyperparams}")
    
    # Compare with default hyperparameters
    print("\n=== COMPARING WITH DEFAULT HYPERPARAMETERS ===")
    
    # Get test_eval data with default hyperparameters
    test_df_default = get_data_for_visualization(
        eval_dataset=dataset,
        experiment=experiment,
        run_name=model,
        exclude_attention=False,
        include_val_eval=False
    )
    test_df_default = apply_main_plot_filters(test_df_default)
    
    # Get test_eval data with CV hyperparameters
    test_df_cv = get_data_for_visualization(
        eval_dataset=dataset,
        experiment=experiment,
        run_name=model,
        exclude_attention=False,
        include_val_eval=False
    )
    
    # Filter to CV-selected hyperparameters
    if 'attention' in best_hyperparams:
        cv_lr = best_hyperparams['attention']['lr']
        cv_wd = best_hyperparams['attention']['weight_decay']
        test_df_cv = test_df_cv[
            (test_df_cv['probe_name'].str.contains('attention', na=False)) &
            (test_df_cv['lr'] == cv_lr) &
            (test_df_cv['weight_decay'] == cv_wd)
        ]
    
    # Compare performance
    if not test_df_default.empty:
        default_attention = test_df_default[test_df_default['probe_name'].str.contains('attention', na=False)]
        if not default_attention.empty:
            print(f"Default attention performance: AUC={default_attention['auc'].mean():.3f}")
    
    if not test_df_cv.empty:
        print(f"CV attention performance: AUC={test_df_cv['auc'].mean():.3f}")
    
    # Show sample-by-sample comparison
    print("\n=== SAMPLE-BY-SAMPLE COMPARISON ===")
    if not test_df_default.empty and not test_df_cv.empty:
        default_by_sample = test_df_default[test_df_default['probe_name'].str.contains('attention', na=False)].groupby('num_positive_samples')['auc'].mean()
        cv_by_sample = test_df_cv.groupby('num_positive_samples')['auc'].mean()
        
        print("Sample Count | Default AUC | CV AUC | Difference")
        print("-------------|-------------|--------|-----------")
        for sample_count in sorted(set(default_by_sample.index) | set(cv_by_sample.index)):
            default_auc = default_by_sample.get(sample_count, np.nan)
            cv_auc = cv_by_sample.get(sample_count, np.nan)
            diff = cv_auc - default_auc if not (np.isnan(default_auc) or np.isnan(cv_auc)) else np.nan
            print(f"{sample_count:11d}  | {default_auc:.3f}      | {cv_auc:.3f}  | {diff:+.3f}")

if __name__ == "__main__":
    debug_attention_cv()



