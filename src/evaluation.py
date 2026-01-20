"""Evaluation module with metrics for time series forecasting models."""

import os
import sys
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# =============================================================================
# Core Metrics
# =============================================================================

def mae(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """Mean Absolute Error - average magnitude of errors."""
    actuals = np.asarray(actuals).flatten()
    predictions = np.asarray(predictions).flatten()
    return np.mean(np.abs(actuals - predictions))


def mse(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """Mean Squared Error - penalizes larger errors more."""
    actuals = np.asarray(actuals).flatten()
    predictions = np.asarray(predictions).flatten()
    return np.mean((actuals - predictions) ** 2)


def rmse(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """Root Mean Squared Error - MSE in original units."""
    return np.sqrt(mse(actuals, predictions))


def mape(actuals: np.ndarray, predictions: np.ndarray, epsilon: float = 1e-10) -> float:
    """Mean Absolute Percentage Error - returns percentage (e.g., 5.2 = 5.2%)."""
    actuals = np.asarray(actuals).flatten()
    predictions = np.asarray(predictions).flatten()
    actuals_safe = np.where(np.abs(actuals) < epsilon, epsilon, actuals)
    return np.mean(np.abs((actuals - predictions) / actuals_safe)) * 100


def smape(actuals: np.ndarray, predictions: np.ndarray, epsilon: float = 1e-10) -> float:
    """Symmetric MAPE - handles zero values better than MAPE."""
    actuals = np.asarray(actuals).flatten()
    predictions = np.asarray(predictions).flatten()
    denominator = (np.abs(actuals) + np.abs(predictions)) / 2
    denominator = np.where(denominator < epsilon, epsilon, denominator)
    return np.mean(np.abs(actuals - predictions) / denominator) * 100


def median_absolute_error(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """Median Absolute Error - robust to outliers."""
    actuals = np.asarray(actuals).flatten()
    predictions = np.asarray(predictions).flatten()
    return np.median(np.abs(actuals - predictions))


def r_squared(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """R-squared (coefficient of determination). Range: -inf to 1, where 1 is perfect."""
    actuals = np.asarray(actuals).flatten()
    predictions = np.asarray(predictions).flatten()
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def directional_accuracy(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """Percentage of correct direction predictions (0-100)."""
    actuals = np.asarray(actuals).flatten()
    predictions = np.asarray(predictions).flatten()
    if len(actuals) < 2:
        return 0.0
    actual_changes = np.diff(actuals)
    pred_changes = np.diff(predictions)
    correct_directions = np.sign(actual_changes) == np.sign(pred_changes)
    return np.mean(correct_directions) * 100


def max_error(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """Maximum absolute error - worst-case prediction error."""
    actuals = np.asarray(actuals).flatten()
    predictions = np.asarray(predictions).flatten()
    return np.max(np.abs(actuals - predictions))


# =============================================================================
# Comprehensive Evaluation
# =============================================================================

def calculate_all_metrics(
    actuals: np.ndarray,
    predictions: np.ndarray,
    include_extended: bool = True
) -> Dict[str, float]:
    """Calculate all available metrics."""
    metrics = {
        'mae': mae(actuals, predictions),
        'rmse': rmse(actuals, predictions),
        'mape': mape(actuals, predictions)
    }
    
    if include_extended:
        metrics.update({
            'mse': mse(actuals, predictions),
            'smape': smape(actuals, predictions),
            'median_ae': median_absolute_error(actuals, predictions),
            'r_squared': r_squared(actuals, predictions),
            'directional_accuracy': directional_accuracy(actuals, predictions),
            'max_error': max_error(actuals, predictions)
        })
    
    return metrics


def evaluate_forecast(
    actuals: Union[np.ndarray, pd.Series],
    predictions: Union[np.ndarray, pd.Series],
    model_name: str = "Model",
    include_extended: bool = False,
    verbose: bool = None
) -> Dict[str, Any]:
    """
    Main evaluation function for forecast predictions.
    
    Returns dict with metrics, predictions, actuals, and errors.
    """
    verbose = verbose if verbose is not None else config.VERBOSE
    
    actuals = np.asarray(actuals).flatten()
    predictions = np.asarray(predictions).flatten()
    
    min_len = min(len(actuals), len(predictions))
    actuals = actuals[:min_len]
    predictions = predictions[:min_len]
    
    errors = actuals - predictions
    metrics = calculate_all_metrics(actuals, predictions, include_extended)
    
    results = {
        'model': model_name,
        **metrics,
        'predictions': predictions,
        'actuals': actuals,
        'errors': errors,
        'n_samples': len(actuals)
    }
    
    if verbose:
        print(f"\n{model_name} Evaluation:")
        print(f"  MAE:  ${metrics['mae']:.2f}")
        print(f"  RMSE: ${metrics['rmse']:.2f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        if include_extended:
            print(f"  R²:   {metrics['r_squared']:.4f}")
            print(f"  Directional Accuracy: {metrics['directional_accuracy']:.1f}%")
    
    return results


# =============================================================================
# Model Comparison
# =============================================================================

def compare_models(
    results_list: List[Dict[str, Any]],
    metrics: List[str] = None,
    sort_by: str = 'mae',
    ascending: bool = True
) -> pd.DataFrame:
    """Compare multiple models, returns sorted DataFrame."""
    if metrics is None:
        metrics = ['mae', 'rmse', 'mape']
    
    comparison_data = []
    for result in results_list:
        row = {'Model': result.get('model', 'Unknown')}
        for metric in metrics:
            if metric in result:
                row[metric.upper()] = result[metric]
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    sort_col = sort_by.upper()
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
    
    return df


def print_comparison_table(
    results_list: List[Dict[str, Any]],
    metrics: List[str] = None,
    sort_by: str = 'mae'
) -> None:
    """Print formatted comparison table to console."""
    if metrics is None:
        metrics = ['mae', 'rmse', 'mape']
    
    sorted_results = sorted(results_list, key=lambda x: x.get(sort_by, float('inf')))
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    header = f"{'Model':<30}"
    for metric in metrics:
        header += f"{metric.upper():>12}"
    print(header)
    print("-" * 70)
    
    for result in sorted_results:
        row = f"{result.get('model', 'Unknown'):<30}"
        for metric in metrics:
            value = result.get(metric, 0)
            if metric == 'mape':
                row += f"{value:>11.2f}%"
            elif metric in ['r_squared', 'directional_accuracy']:
                row += f"{value:>12.2f}"
            else:
                row += f"${value:>11.2f}"
        print(row)
    
    best_model = sorted_results[0].get('model', 'Unknown')
    print("-" * 70)
    print(f"Best Model (by {sort_by.upper()}): {best_model}")


def rank_models(
    results_list: List[Dict[str, Any]],
    metrics: List[str] = None,
    weights: Dict[str, float] = None
) -> pd.DataFrame:
    """Rank models using weighted composite score."""
    if metrics is None:
        metrics = ['mae', 'rmse', 'mape']
    
    if weights is None:
        weights = {m: 1.0 / len(metrics) for m in metrics}
    
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    model_names = [r.get('model', f'Model_{i}') for i, r in enumerate(results_list)]
    metric_values = {m: [r.get(m, float('inf')) for r in results_list] for m in metrics}
    
    # Normalize metrics (min-max scaling)
    normalized = {}
    for metric, values in metric_values.items():
        min_val, max_val = min(values), max(values)
        if max_val > min_val:
            normalized[metric] = [(v - min_val) / (max_val - min_val) for v in values]
        else:
            normalized[metric] = [0.0] * len(values)
    
    composite_scores = []
    for i in range(len(results_list)):
        score = sum(weights.get(m, 0) * normalized[m][i] for m in metrics)
        composite_scores.append(score)
    
    df = pd.DataFrame({'Model': model_names, 'Composite Score': composite_scores})
    for metric in metrics:
        df[metric.upper()] = metric_values[metric]
    
    df = df.sort_values('Composite Score').reset_index(drop=True)
    df['Rank'] = range(1, len(df) + 1)
    
    cols = ['Rank', 'Model', 'Composite Score'] + [m.upper() for m in metrics]
    return df[cols]


# =============================================================================
# Error Analysis
# =============================================================================

def analyze_errors(
    actuals: np.ndarray,
    predictions: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None
) -> Dict[str, Any]:
    """Detailed error analysis with statistics and percentiles."""
    actuals = np.asarray(actuals).flatten()
    predictions = np.asarray(predictions).flatten()
    errors = actuals - predictions
    abs_errors = np.abs(errors)
    pct_errors = np.abs(errors / actuals) * 100
    
    analysis = {
        'error_mean': np.mean(errors),
        'error_std': np.std(errors),
        'error_median': np.median(errors),
        'error_skewness': _calculate_skewness(errors),
        'error_kurtosis': _calculate_kurtosis(errors),
        'abs_error_mean': np.mean(abs_errors),
        'abs_error_std': np.std(abs_errors),
        'abs_error_median': np.median(abs_errors),
        'pct_error_mean': np.mean(pct_errors),
        'pct_error_std': np.std(pct_errors),
        'max_overestimate': np.max(-errors),
        'max_underestimate': np.max(errors),
        'n_overestimates': np.sum(predictions > actuals),
        'n_underestimates': np.sum(predictions < actuals),
        'error_percentiles': {
            '5th': np.percentile(abs_errors, 5),
            '25th': np.percentile(abs_errors, 25),
            '50th': np.percentile(abs_errors, 50),
            '75th': np.percentile(abs_errors, 75),
            '95th': np.percentile(abs_errors, 95)
        }
    }
    
    if dates is not None and len(dates) == len(errors):
        analysis['temporal'] = _analyze_temporal_errors(errors, dates)
    
    return analysis


def _calculate_skewness(data: np.ndarray) -> float:
    """Calculate skewness."""
    n = len(data)
    if n < 3:
        return 0.0
    mean, std = np.mean(data), np.std(data, ddof=1)
    if std == 0:
        return 0.0
    return np.mean(((data - mean) / std) ** 3)


def _calculate_kurtosis(data: np.ndarray) -> float:
    """Calculate excess kurtosis."""
    n = len(data)
    if n < 4:
        return 0.0
    mean, std = np.mean(data), np.std(data, ddof=1)
    if std == 0:
        return 0.0
    return np.mean(((data - mean) / std) ** 4) - 3


def _analyze_temporal_errors(errors: np.ndarray, dates: pd.DatetimeIndex) -> Dict[str, Any]:
    """Analyze errors by day of week and month."""
    df = pd.DataFrame({'error': errors, 'abs_error': np.abs(errors)}, index=dates)
    temporal = {}
    
    if hasattr(df.index, 'dayofweek'):
        dow_errors = df.groupby(df.index.dayofweek)['abs_error'].mean()
        temporal['day_of_week'] = dow_errors.to_dict()
    
    if hasattr(df.index, 'month'):
        monthly_errors = df.groupby(df.index.month)['abs_error'].mean()
        temporal['month'] = monthly_errors.to_dict()
    
    if len(errors) > 10:
        x = np.arange(len(errors))
        slope = np.polyfit(x, np.abs(errors), 1)[0]
        temporal['error_trend_slope'] = slope
    
    return temporal


# =============================================================================
# Visualization Support
# =============================================================================

def get_error_distribution_data(
    actuals: np.ndarray,
    predictions: np.ndarray,
    bins: int = 30
) -> Dict[str, np.ndarray]:
    """Get histogram data for error distribution visualization."""
    errors = np.asarray(actuals).flatten() - np.asarray(predictions).flatten()
    hist, bin_edges = np.histogram(errors, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return {
        'errors': errors,
        'histogram': hist,
        'bin_edges': bin_edges,
        'bin_centers': bin_centers
    }


# =============================================================================
# Reporting
# =============================================================================

def generate_evaluation_report(
    results_list: List[Dict[str, Any]],
    output_path: Optional[str] = None
) -> str:
    """Generate text report comparing all models."""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("TIME SERIES FORECASTING - EVALUATION REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append("MODEL COMPARISON (sorted by MAE)")
    report_lines.append("-" * 70)
    
    sorted_results = sorted(results_list, key=lambda x: x.get('mae', float('inf')))
    
    header = f"{'Rank':<6}{'Model':<30}{'MAE':>10}{'RMSE':>10}{'MAPE':>10}"
    report_lines.append(header)
    report_lines.append("-" * 70)
    
    for i, result in enumerate(sorted_results, 1):
        row = f"{i:<6}{result.get('model', 'Unknown'):<30}"
        row += f"${result.get('mae', 0):>9.2f}"
        row += f"${result.get('rmse', 0):>9.2f}"
        row += f"{result.get('mape', 0):>9.2f}%"
        report_lines.append(row)
    
    report_lines.append("")
    report_lines.append(f"Best Model: {sorted_results[0].get('model', 'Unknown')}")
    report_lines.append(f"  MAE:  ${sorted_results[0].get('mae', 0):.2f}")
    report_lines.append(f"  RMSE: ${sorted_results[0].get('rmse', 0):.2f}")
    report_lines.append(f"  MAPE: {sorted_results[0].get('mape', 0):.2f}%")
    
    report = "\n".join(report_lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        if config.VERBOSE:
            print(f"Report saved to: {output_path}")
    
    return report


if __name__ == "__main__":
    print("=" * 60)
    print("Evaluation Module - Example Usage")
    print("=" * 60)
    
    np.random.seed(42)
    n = 100
    actuals = 100 + np.cumsum(np.random.randn(n) * 2)
    predictions = actuals + np.random.randn(n) * 5
    
    print("\n--- Individual Metrics ---")
    print(f"MAE:  ${mae(actuals, predictions):.2f}")
    print(f"RMSE: ${rmse(actuals, predictions):.2f}")
    print(f"MAPE: {mape(actuals, predictions):.2f}%")
    print(f"R²:   {r_squared(actuals, predictions):.4f}")
    print(f"Directional Accuracy: {directional_accuracy(actuals, predictions):.1f}%")
    
    print("\n--- Full Evaluation ---")
    results = evaluate_forecast(actuals, predictions, model_name="Test Model", include_extended=True)
    
    print("\n--- Model Comparison ---")
    results_list = [
        evaluate_forecast(actuals, predictions + np.random.randn(n) * 3, "Model A", verbose=False),
        evaluate_forecast(actuals, predictions + np.random.randn(n) * 5, "Model B", verbose=False),
        evaluate_forecast(actuals, predictions + np.random.randn(n) * 7, "Model C", verbose=False),
    ]
    
    print_comparison_table(results_list)
    
    print("\n--- Model Ranking ---")
    ranking = rank_models(results_list)
    print(ranking.to_string(index=False))
    
    print("\n--- Error Analysis ---")
    error_analysis = analyze_errors(actuals, predictions)
    print(f"Mean Error: ${error_analysis['error_mean']:.2f}")
    print(f"Error Std:  ${error_analysis['error_std']:.2f}")
    print(f"Overestimates:   {error_analysis['n_overestimates']}")
    print(f"Underestimates:  {error_analysis['n_underestimates']}")
