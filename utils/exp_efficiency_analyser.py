import pandas as pd
import numpy as np
import re

ERROR_METRICS = ['mse', 'mae']
EFFICIENCY_METRICS = ['seq_per_second_train', 'peak_memory_train',
                      'seq_per_second_test', 'peak_memory_test', 'flops', 'params']
CONFIG = {
    'error_metric': 'mse',
    'sel_efficiency_metrics': ['flops'],
    'baseline_model': 'DLinear',
    'w': -0.07
}
COMPOSITE_CONFIGS = [
    ['flops'],
    ['params'],
    ['seq_per_second_train', 'peak_memory_train'],
    ['seq_per_second_test', 'peak_memory_test']
]


def parse_results(file_path, datasets, forecast_horizons, metric):
    results_mean = {}
    results_std = {}
    
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]
    
    for horizon in forecast_horizons:
        horizon_data_mean = {}
        horizon_data_std = {}
        for dataset in datasets:
            scores_dict = {}
            for i in range(0, len(lines), 2):
                model_line = lines[i]
                metrics_line = lines[i + 1]
                exact_pattern = f"{dataset}_{horizon}"
                
                if exact_pattern in model_line:
                    parts = model_line.split('_')
                    model_name = next(parts[j+2] for j, part in enumerate(parts) if part == dataset)
                    
                    pattern = fr"{metric}:([\d.]+)"
                    match = re.search(pattern, metrics_line)
                    score = float(match.group(1)) if match else None
                    
                    if score is not None:
                        if model_name not in scores_dict:
                            scores_dict[model_name] = []
                        scores_dict[model_name].append(score)
            
            if scores_dict:
                horizon_data_mean[dataset] = {k: np.mean(v) for k, v in scores_dict.items()}
                horizon_data_std[dataset] = {k: np.std(v) if len(v) > 1 else 0 for k, v in scores_dict.items()}
        
        results_mean[f'H{horizon}'] = pd.DataFrame.from_dict(horizon_data_mean, orient='index')
        results_std[f'H{horizon}'] = pd.DataFrame.from_dict(horizon_data_std, orient='index')
    
    return (pd.concat({k: v for k, v in results_mean.items()}, axis=1),
            pd.concat({k: v for k, v in results_std.items()}, axis=1))


def compute_composite_metric(performance, error_metric, 
                            sel_efficiency_metrics, 
                            baseline_model='DLinear',
                            w=1.0):
    """
    Compute composite metric with configurable exponents.
    
    Args:
        performance: Dictionary of DataFrames with metrics
        error_metric: Name of error metric (lower is better)
        sel_efficiency_metrics: List of efficiency metrics to include
        baseline_model: Name of baseline model
        w: Exponent weight (default=1.0)
    
    Returns:
        pandas.Series with composite scores
    """
    # Get baseline values
    baseline_error = performance[error_metric].groupby(level=1, axis=1).mean().mean()[baseline_model]
    
    # Define metric directions and initialize baseline values
    efficiency_config = {
        'seq_per_second_train': {'sign': 1},   # Higher better
        'peak_memory_train': {'sign': -1},     # Lower better
        'seq_per_second_test': {'sign': 1},    # Higher better
        'peak_memory_test': {'sign': -1},      # Lower better
        'flops': {'sign': -1},                 # Lower better
        'params': {'sign': -1}                 # Lower better
    }
    
    # Store baseline efficiency values
    for metric in sel_efficiency_metrics:
        efficiency_config[metric]['baseline'] = performance[metric].groupby(level=1, axis=1).mean().mean()[baseline_model]
    
    # Get all models
    models = performance[error_metric].columns.get_level_values(1).unique()
    composite_scores = pd.Series(index=models, dtype=float)
    
    for model in models:
        if model == baseline_model:
            composite_scores[model] = 1.0
            continue
            
        # Error component
        model_error = performance[error_metric].groupby(level=1, axis=1).mean().mean()[model]
        error_ratio = baseline_error / model_error
        
        # Efficiency components
        efficiency_ratio = 1.0
        for metric in sel_efficiency_metrics:
            cfg = efficiency_config[metric]
            model_efficiency = performance[metric].groupby(level=1, axis=1).mean().mean()[model]
            baseline_efficiency = cfg['baseline']
            
            # Calculate ratio with signed exponent
            ratio = (baseline_efficiency / model_efficiency) ** (cfg['sign'] * (w))
            efficiency_ratio *= ratio            

        composite_scores[model] = error_ratio * efficiency_ratio
    
    return composite_scores.sort_values(ascending=False)


def scale_performance_metrics(performance):
    scaled = {}
    for key, df in performance.items():
        if 'params' in key:
            scaled[key] = df / 1e6  # scale to millions
        elif 'flops' in key:
            scaled[key] = df / 1e9  # scale to giga
        elif 'peak_memory' in key:
            scaled[key] = df / (1024**2)  # bytes to megabytes
        else:
            scaled[key] = df.copy()
    return scaled


def compute_avg_metric(performance, metric):
    return performance[metric].groupby(level=1, axis=1).mean().mean()


def analyze_efficiency_experiments(path_results_file,
                                   path_efficiency_file,
                                   config_df):
    # Extract datasets and horizons
    datasets = config_df.index.tolist()
    forecast_horizons = [int(col[0][1:]) for col in config_df.columns]    

    # Parse results files
    performance = {}
    for metric in ERROR_METRICS:
        mean_metric, _ = parse_results(path_results_file, datasets, forecast_horizons, metric=metric)
        performance[metric] = mean_metric
    for metric in EFFICIENCY_METRICS:
        mean_eff, _ = parse_results(path_efficiency_file, datasets, forecast_horizons, metric=metric)
        performance[metric] = mean_eff
    
    # Scale performance dictionary
    scaled_performance = scale_performance_metrics(performance)
    
    composite_scores = {}
    for metrics in COMPOSITE_CONFIGS:
        CONFIG.update({'sel_efficiency_metrics': metrics})
        key = " + ".join(metrics)
        composite_scores[key] = compute_composite_metric(
            performance=scaled_performance,
            **CONFIG
        )
    
    # Collect average performance metrics
    all_metrics = list(performance.keys())

    rows = {}
    for metric in all_metrics:
        rows[metric] = compute_avg_metric(scaled_performance, metric)

    # Add composite metrics to rows
    for name, series in composite_scores.items():
        rows[f"{name} (c)"] = series

    # Build DataFrame: rows are metrics, columns are models
    metrics_table = pd.DataFrame(rows).T
    
    print("\nMetrics Table (rows = metrics, columns = models):")
    print(metrics_table.round(2))
    print()
    analysis_file = f"{path_efficiency_file.split('.')[0]}_analysis.txt"
    metrics_table.round(2).to_csv(analysis_file, sep=',', index=True)
    
    return analysis_file
    