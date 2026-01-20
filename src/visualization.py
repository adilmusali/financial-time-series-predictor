"""Visualization module for time series forecasting predictions and comparisons."""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, Dict, Any, List, Tuple, Union
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# =============================================================================
# Plot Configuration
# =============================================================================

def setup_plot_style():
    """Set up matplotlib style for consistent plots."""
    try:
        plt.style.use(config.PLOT_STYLE)
    except OSError:
        plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.rcParams.update({
        'figure.figsize': config.FIGURE_SIZE,
        'figure.dpi': 100,
        'savefig.dpi': config.PLOT_DPI,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 1.5,
    })


# =============================================================================
# Single Model Plots
# =============================================================================

def plot_predictions(
    actuals: Union[np.ndarray, pd.Series],
    predictions: Union[np.ndarray, pd.Series],
    dates: Optional[pd.DatetimeIndex] = None,
    model_name: str = "Model",
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot actual vs predicted values for a single model.
    """
    setup_plot_style()
    figsize = figsize or config.FIGURE_SIZE
    
    actuals = np.asarray(actuals).flatten()
    predictions = np.asarray(predictions).flatten()
    
    min_len = min(len(actuals), len(predictions))
    actuals = actuals[:min_len]
    predictions = predictions[:min_len]
    
    if dates is not None:
        dates = dates[:min_len]
        x_axis = dates
    else:
        x_axis = np.arange(min_len)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(x_axis, actuals, label='Actual', color='#2E86AB', linewidth=2)
    ax.plot(x_axis, predictions, label=f'{model_name} Prediction', 
            color='#E94F37', linewidth=2, linestyle='--')
    
    ax.set_xlabel('Date' if dates is not None else 'Time Step')
    ax.set_ylabel('Price ($)')
    ax.set_title(title or f'{model_name}: Actual vs Predicted')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    if dates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        _save_figure(fig, save_path)
    
    if show_plot:
        plt.show()
    
    return fig


def plot_predictions_with_confidence(
    actuals: Union[np.ndarray, pd.Series],
    predictions: Union[np.ndarray, pd.Series],
    lower_bound: Union[np.ndarray, pd.Series],
    upper_bound: Union[np.ndarray, pd.Series],
    dates: Optional[pd.DatetimeIndex] = None,
    model_name: str = "Model",
    confidence_level: float = 0.95,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """Plot predictions with confidence intervals."""
    setup_plot_style()
    figsize = figsize or config.FIGURE_SIZE
    
    actuals = np.asarray(actuals).flatten()
    predictions = np.asarray(predictions).flatten()
    lower_bound = np.asarray(lower_bound).flatten()
    upper_bound = np.asarray(upper_bound).flatten()
    
    min_len = min(len(actuals), len(predictions), len(lower_bound), len(upper_bound))
    actuals = actuals[:min_len]
    predictions = predictions[:min_len]
    lower_bound = lower_bound[:min_len]
    upper_bound = upper_bound[:min_len]
    
    if dates is not None:
        dates = dates[:min_len]
        x_axis = dates
    else:
        x_axis = np.arange(min_len)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.fill_between(x_axis, lower_bound, upper_bound, 
                    alpha=0.3, color='#E94F37', 
                    label=f'{int(confidence_level*100)}% Confidence Interval')
    ax.plot(x_axis, actuals, label='Actual', color='#2E86AB', linewidth=2)
    ax.plot(x_axis, predictions, label=f'{model_name} Prediction', 
            color='#E94F37', linewidth=2, linestyle='--')
    
    ax.set_xlabel('Date' if dates is not None else 'Time Step')
    ax.set_ylabel('Price ($)')
    ax.set_title(f'{model_name}: Predictions with {int(confidence_level*100)}% Confidence Interval')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    if dates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        _save_figure(fig, save_path)
    
    if show_plot:
        plt.show()
    
    return fig


# =============================================================================
# Model Comparison Plots
# =============================================================================

def plot_model_comparison(
    actuals: Union[np.ndarray, pd.Series],
    predictions_dict: Dict[str, np.ndarray],
    dates: Optional[pd.DatetimeIndex] = None,
    title: str = "Model Comparison: Actual vs Predictions",
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Compare predictions from multiple models on the same plot.
    """
    setup_plot_style()
    figsize = figsize or (config.FIGURE_SIZE[0], config.FIGURE_SIZE[1] * 1.2)
    
    actuals = np.asarray(actuals).flatten()
    
    # Get minimum length across all predictions
    min_len = len(actuals)
    for preds in predictions_dict.values():
        min_len = min(min_len, len(np.asarray(preds).flatten()))
    
    actuals = actuals[:min_len]
    
    if dates is not None:
        dates = dates[:min_len]
        x_axis = dates
    else:
        x_axis = np.arange(min_len)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color palette for models
    colors = ['#E94F37', '#44AF69', '#F8333C', '#FCAB10', '#2B9EB3', '#8E44AD']
    linestyles = ['--', '-.', ':', '--', '-.', ':']
    
    # Plot actual values
    ax.plot(x_axis, actuals, label='Actual', color='#2E86AB', linewidth=2.5)
    
    # Plot each model's predictions
    for i, (model_name, predictions) in enumerate(predictions_dict.items()):
        predictions = np.asarray(predictions).flatten()[:min_len]
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        ax.plot(x_axis, predictions, label=model_name, 
                color=color, linewidth=1.5, linestyle=linestyle)
    
    ax.set_xlabel('Date' if dates is not None else 'Time Step')
    ax.set_ylabel('Price ($)')
    ax.set_title(title)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    if dates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        _save_figure(fig, save_path)
    
    if show_plot:
        plt.show()
    
    return fig


def plot_metrics_comparison(
    results_list: List[Dict[str, Any]],
    metrics: List[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Bar chart comparing metrics across multiple models.
    """
    setup_plot_style()
    
    if metrics is None:
        metrics = ['mae', 'rmse', 'mape']
    
    n_metrics = len(metrics)
    n_models = len(results_list)
    
    figsize = figsize or (max(10, n_models * 2), 6)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))
    model_names = [r.get('model', f'Model {i+1}') for i, r in enumerate(results_list)]
    
    for ax, metric in zip(axes, metrics):
        values = [r.get(metric, 0) for r in results_list]
        bars = ax.bar(model_names, values, color=colors)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            label = f'{val:.2f}%' if metric == 'mape' else f'${val:.2f}'
            ax.annotate(label,
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} by Model')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Model Performance Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        _save_figure(fig, save_path)
    
    if show_plot:
        plt.show()
    
    return fig


def plot_metrics_heatmap(
    results_list: List[Dict[str, Any]],
    metrics: List[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """Heatmap of normalized metrics for model comparison."""
    setup_plot_style()
    
    if metrics is None:
        metrics = ['mae', 'rmse', 'mape']
    
    figsize = figsize or (10, 6)
    
    model_names = [r.get('model', f'Model {i+1}') for i, r in enumerate(results_list)]
    
    # Build data matrix
    data = np.zeros((len(results_list), len(metrics)))
    for i, result in enumerate(results_list):
        for j, metric in enumerate(metrics):
            data[i, j] = result.get(metric, np.nan)
    
    # Normalize for visualization (lower is better, so invert)
    normalized = np.zeros_like(data)
    for j in range(len(metrics)):
        col = data[:, j]
        min_val, max_val = col.min(), col.max()
        if max_val > min_val:
            normalized[:, j] = 1 - (col - min_val) / (max_val - min_val)  # Invert so higher = better
        else:
            normalized[:, j] = 0.5
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(normalized, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_yticklabels(model_names)
    
    # Add value annotations
    for i in range(len(model_names)):
        for j in range(len(metrics)):
            val = data[i, j]
            text = f'{val:.2f}%' if metrics[j] == 'mape' else f'{val:.2f}'
            color = 'white' if normalized[i, j] < 0.3 or normalized[i, j] > 0.7 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=10)
    
    ax.set_title('Model Performance Heatmap\n(Green = Better Performance)')
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Relative Performance', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    if save_path:
        _save_figure(fig, save_path)
    
    if show_plot:
        plt.show()
    
    return fig


# =============================================================================
# Error Analysis Plots
# =============================================================================

def plot_error_distribution(
    actuals: Union[np.ndarray, pd.Series],
    predictions: Union[np.ndarray, pd.Series],
    model_name: str = "Model",
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """Plot error distribution with histogram and statistics."""
    setup_plot_style()
    figsize = figsize or (12, 5)
    
    actuals = np.asarray(actuals).flatten()
    predictions = np.asarray(predictions).flatten()
    
    min_len = min(len(actuals), len(predictions))
    errors = actuals[:min_len] - predictions[:min_len]
    pct_errors = (errors / actuals[:min_len]) * 100
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Absolute error distribution
    axes[0].hist(errors, bins=30, edgecolor='black', alpha=0.7, color='#2E86AB')
    axes[0].axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: ${np.mean(errors):.2f}')
    axes[0].axvline(np.median(errors), color='green', linestyle=':', linewidth=2,
                    label=f'Median: ${np.median(errors):.2f}')
    axes[0].axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    axes[0].set_xlabel('Prediction Error ($)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{model_name}: Error Distribution')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Percentage error distribution
    axes[1].hist(pct_errors, bins=30, edgecolor='black', alpha=0.7, color='#44AF69')
    axes[1].axvline(np.mean(pct_errors), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(pct_errors):.2f}%')
    axes[1].axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    axes[1].set_xlabel('Percentage Error (%)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{model_name}: Percentage Error Distribution')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        _save_figure(fig, save_path)
    
    if show_plot:
        plt.show()
    
    return fig


def plot_residuals(
    actuals: Union[np.ndarray, pd.Series],
    predictions: Union[np.ndarray, pd.Series],
    dates: Optional[pd.DatetimeIndex] = None,
    model_name: str = "Model",
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """Plot residuals analysis (residuals over time and scatter)."""
    setup_plot_style()
    figsize = figsize or (12, 8)
    
    actuals = np.asarray(actuals).flatten()
    predictions = np.asarray(predictions).flatten()
    
    min_len = min(len(actuals), len(predictions))
    actuals = actuals[:min_len]
    predictions = predictions[:min_len]
    residuals = actuals - predictions
    
    if dates is not None:
        dates = dates[:min_len]
        x_axis = dates
    else:
        x_axis = np.arange(min_len)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Residuals over time
    axes[0, 0].plot(x_axis, residuals, color='#2E86AB', linewidth=1)
    axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[0, 0].fill_between(x_axis, residuals, 0, alpha=0.3, color='#2E86AB')
    axes[0, 0].set_xlabel('Date' if dates is not None else 'Time Step')
    axes[0, 0].set_ylabel('Residual ($)')
    axes[0, 0].set_title('Residuals Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    if dates is not None:
        axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        for label in axes[0, 0].get_xticklabels():
            label.set_rotation(45)
    
    # Predicted vs Actual scatter
    axes[0, 1].scatter(actuals, predictions, alpha=0.5, color='#44AF69', s=20)
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
    axes[0, 1].set_xlabel('Actual ($)')
    axes[0, 1].set_ylabel('Predicted ($)')
    axes[0, 1].set_title('Predicted vs Actual')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Residuals vs Predicted
    axes[1, 0].scatter(predictions, residuals, alpha=0.5, color='#E94F37', s=20)
    axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1, 0].set_xlabel('Predicted ($)')
    axes[1, 0].set_ylabel('Residual ($)')
    axes[1, 0].set_title('Residuals vs Predicted')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals histogram
    axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='#FCAB10')
    axes[1, 1].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[1, 1].set_xlabel('Residual ($)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Residual Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name}: Residual Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        _save_figure(fig, save_path)
    
    if show_plot:
        plt.show()
    
    return fig


# =============================================================================
# Forecast Plots
# =============================================================================

def plot_forecast(
    historical: Union[np.ndarray, pd.Series],
    forecast: Union[np.ndarray, pd.Series],
    historical_dates: Optional[pd.DatetimeIndex] = None,
    forecast_dates: Optional[pd.DatetimeIndex] = None,
    lower_bound: Optional[np.ndarray] = None,
    upper_bound: Optional[np.ndarray] = None,
    model_name: str = "Model",
    n_historical: int = 60,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot historical data with future forecast.
    """
    setup_plot_style()
    figsize = figsize or (14, 6)
    
    historical = np.asarray(historical).flatten()
    forecast = np.asarray(forecast).flatten()
    
    # Limit historical data shown
    if len(historical) > n_historical:
        historical = historical[-n_historical:]
        if historical_dates is not None:
            historical_dates = historical_dates[-n_historical:]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create x-axis
    if historical_dates is not None and forecast_dates is not None:
        hist_x = historical_dates
        fore_x = forecast_dates
    else:
        hist_x = np.arange(len(historical))
        fore_x = np.arange(len(historical), len(historical) + len(forecast))
    
    # Plot historical
    ax.plot(hist_x, historical, label='Historical', color='#2E86AB', linewidth=2)
    
    # Plot forecast
    ax.plot(fore_x, forecast, label=f'{model_name} Forecast', 
            color='#E94F37', linewidth=2, linestyle='--')
    
    # Plot confidence interval if provided
    if lower_bound is not None and upper_bound is not None:
        lower_bound = np.asarray(lower_bound).flatten()
        upper_bound = np.asarray(upper_bound).flatten()
        ax.fill_between(fore_x, lower_bound, upper_bound, 
                       alpha=0.3, color='#E94F37', label='Confidence Interval')
    
    # Mark the transition point
    if historical_dates is not None:
        ax.axvline(historical_dates[-1], color='gray', linestyle=':', alpha=0.7)
    else:
        ax.axvline(len(historical) - 0.5, color='gray', linestyle=':', alpha=0.7)
    
    ax.set_xlabel('Date' if historical_dates is not None else 'Time Step')
    ax.set_ylabel('Price ($)')
    ax.set_title(f'{model_name}: Forecast')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    if historical_dates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        _save_figure(fig, save_path)
    
    if show_plot:
        plt.show()
    
    return fig


def plot_multi_model_forecast(
    historical: Union[np.ndarray, pd.Series],
    forecasts_dict: Dict[str, np.ndarray],
    historical_dates: Optional[pd.DatetimeIndex] = None,
    forecast_dates: Optional[pd.DatetimeIndex] = None,
    n_historical: int = 60,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """Plot forecasts from multiple models together."""
    setup_plot_style()
    figsize = figsize or (14, 6)
    
    historical = np.asarray(historical).flatten()
    
    if len(historical) > n_historical:
        historical = historical[-n_historical:]
        if historical_dates is not None:
            historical_dates = historical_dates[-n_historical:]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#E94F37', '#44AF69', '#FCAB10', '#8E44AD', '#2B9EB3']
    linestyles = ['--', '-.', ':', '--', '-.']
    
    # Create x-axis
    if historical_dates is not None:
        hist_x = historical_dates
    else:
        hist_x = np.arange(len(historical))
    
    # Plot historical
    ax.plot(hist_x, historical, label='Historical', color='#2E86AB', linewidth=2.5)
    
    # Plot each forecast
    for i, (model_name, forecast) in enumerate(forecasts_dict.items()):
        forecast = np.asarray(forecast).flatten()
        
        if forecast_dates is not None:
            fore_x = forecast_dates[:len(forecast)]
        else:
            fore_x = np.arange(len(historical), len(historical) + len(forecast))
        
        ax.plot(fore_x, forecast, label=f'{model_name}', 
                color=colors[i % len(colors)], linewidth=1.5, 
                linestyle=linestyles[i % len(linestyles)])
    
    # Mark transition point
    if historical_dates is not None:
        ax.axvline(historical_dates[-1], color='gray', linestyle=':', alpha=0.7)
    else:
        ax.axvline(len(historical) - 0.5, color='gray', linestyle=':', alpha=0.7)
    
    ax.set_xlabel('Date' if historical_dates is not None else 'Time Step')
    ax.set_ylabel('Price ($)')
    ax.set_title('Multi-Model Forecast Comparison')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    if historical_dates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        _save_figure(fig, save_path)
    
    if show_plot:
        plt.show()
    
    return fig


# =============================================================================
# Train/Test Split Visualization
# =============================================================================

def plot_train_test_split(
    train_data: Union[np.ndarray, pd.Series],
    test_data: Union[np.ndarray, pd.Series],
    train_dates: Optional[pd.DatetimeIndex] = None,
    test_dates: Optional[pd.DatetimeIndex] = None,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """Visualize train/test split."""
    setup_plot_style()
    figsize = figsize or config.FIGURE_SIZE
    
    train_data = np.asarray(train_data).flatten()
    test_data = np.asarray(test_data).flatten()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if train_dates is not None and test_dates is not None:
        ax.plot(train_dates, train_data, label='Training Data', color='#2E86AB', linewidth=1.5)
        ax.plot(test_dates, test_data, label='Test Data', color='#E94F37', linewidth=1.5)
        ax.axvline(train_dates[-1], color='gray', linestyle='--', linewidth=2, 
                  label='Train/Test Split')
    else:
        train_x = np.arange(len(train_data))
        test_x = np.arange(len(train_data), len(train_data) + len(test_data))
        ax.plot(train_x, train_data, label='Training Data', color='#2E86AB', linewidth=1.5)
        ax.plot(test_x, test_data, label='Test Data', color='#E94F37', linewidth=1.5)
        ax.axvline(len(train_data) - 0.5, color='gray', linestyle='--', linewidth=2,
                  label='Train/Test Split')
    
    ax.set_xlabel('Date' if train_dates is not None else 'Time Step')
    ax.set_ylabel('Price ($)')
    ax.set_title(f'Train/Test Split ({len(train_data)} train, {len(test_data)} test)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    if train_dates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        _save_figure(fig, save_path)
    
    if show_plot:
        plt.show()
    
    return fig


# =============================================================================
# Summary Dashboard
# =============================================================================

def plot_evaluation_dashboard(
    actuals: Union[np.ndarray, pd.Series],
    predictions_dict: Dict[str, np.ndarray],
    results_list: List[Dict[str, Any]],
    dates: Optional[pd.DatetimeIndex] = None,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Comprehensive evaluation dashboard with multiple views.
    
    Args:
        actuals: Actual values
        predictions_dict: Dictionary of model predictions
        results_list: List of evaluation results
        dates: Optional datetime index
        figsize: Figure size
        save_path: Path to save
        show_plot: Whether to display
    
    Returns:
        matplotlib Figure
    """
    setup_plot_style()
    figsize = figsize or (16, 12)
    
    actuals = np.asarray(actuals).flatten()
    
    fig = plt.figure(figsize=figsize)
    
    # Create grid
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Model Comparison (top left)
    ax1 = fig.add_subplot(gs[0, :])
    min_len = len(actuals)
    for preds in predictions_dict.values():
        min_len = min(min_len, len(np.asarray(preds).flatten()))
    
    if dates is not None:
        x_axis = dates[:min_len]
    else:
        x_axis = np.arange(min_len)
    
    ax1.plot(x_axis, actuals[:min_len], label='Actual', color='#2E86AB', linewidth=2)
    colors = ['#E94F37', '#44AF69', '#FCAB10', '#8E44AD']
    for i, (name, preds) in enumerate(predictions_dict.items()):
        preds = np.asarray(preds).flatten()[:min_len]
        ax1.plot(x_axis, preds, label=name, color=colors[i % len(colors)], 
                linewidth=1.5, linestyle='--')
    ax1.set_xlabel('Date' if dates is not None else 'Time Step')
    ax1.set_ylabel('Price ($)')
    ax1.set_title('All Models: Actual vs Predicted')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Metrics Bar Chart (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    metrics = ['mae', 'rmse']
    model_names = [r.get('model', f'M{i+1}') for i, r in enumerate(results_list)]
    x = np.arange(len(model_names))
    width = 0.35
    
    mae_vals = [r.get('mae', 0) for r in results_list]
    rmse_vals = [r.get('rmse', 0) for r in results_list]
    
    bars1 = ax2.bar(x - width/2, mae_vals, width, label='MAE', color='#2E86AB')
    bars2 = ax2.bar(x + width/2, rmse_vals, width, label='RMSE', color='#E94F37')
    ax2.set_ylabel('Error ($)')
    ax2.set_title('MAE and RMSE Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. MAPE Bar Chart (middle right)
    ax3 = fig.add_subplot(gs[1, 1])
    mape_vals = [r.get('mape', 0) for r in results_list]
    bars = ax3.bar(model_names, mape_vals, color='#44AF69')
    for bar, val in zip(bars, mape_vals):
        ax3.annotate(f'{val:.2f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    ax3.set_ylabel('MAPE (%)')
    ax3.set_title('MAPE Comparison')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Error Distribution (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    for i, (name, preds) in enumerate(predictions_dict.items()):
        preds = np.asarray(preds).flatten()[:min_len]
        errors = actuals[:min_len] - preds
        ax4.hist(errors, bins=20, alpha=0.5, label=name, color=colors[i % len(colors)])
    ax4.axvline(0, color='black', linestyle='--', linewidth=1)
    ax4.set_xlabel('Error ($)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Error Distributions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Model Ranking Table (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Create ranking table
    sorted_results = sorted(results_list, key=lambda x: x.get('mae', float('inf')))
    table_data = []
    for i, r in enumerate(sorted_results, 1):
        table_data.append([
            i,
            r.get('model', 'Unknown'),
            f"${r.get('mae', 0):.2f}",
            f"${r.get('rmse', 0):.2f}",
            f"{r.get('mape', 0):.2f}%"
        ])
    
    table = ax5.table(
        cellText=table_data,
        colLabels=['Rank', 'Model', 'MAE', 'RMSE', 'MAPE'],
        loc='center',
        cellLoc='center',
        colColours=['#E8E8E8'] * 5
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax5.set_title('Model Ranking (by MAE)', fontsize=12, pad=20)
    
    plt.suptitle('Model Evaluation Dashboard', fontsize=16, y=0.98)
    
    if save_path:
        _save_figure(fig, save_path)
    
    if show_plot:
        plt.show()
    
    return fig


# =============================================================================
# Utility Functions
# =============================================================================

def _save_figure(fig: plt.Figure, save_path: str) -> None:
    """Save figure to file, creating directories if needed."""
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight', facecolor='white')
    if config.VERBOSE:
        print(f"Plot saved to: {save_path}")


def save_all_plots(
    actuals: Union[np.ndarray, pd.Series],
    predictions_dict: Dict[str, np.ndarray],
    results_list: List[Dict[str, Any]],
    dates: Optional[pd.DatetimeIndex] = None,
    output_dir: Optional[str] = None,
    show_plots: bool = False
) -> Dict[str, str]:
    """
    Generate and save all standard plots.
    
    Returns dictionary mapping plot names to file paths.
    """
    output_dir = output_dir or config.PLOTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    saved_plots = {}
    
    # Model comparison
    path = os.path.join(output_dir, 'model_comparison.png')
    plot_model_comparison(actuals, predictions_dict, dates, 
                         save_path=path, show_plot=show_plots)
    saved_plots['model_comparison'] = path
    
    # Metrics comparison
    path = os.path.join(output_dir, 'metrics_comparison.png')
    plot_metrics_comparison(results_list, save_path=path, show_plot=show_plots)
    saved_plots['metrics_comparison'] = path
    
    # Metrics heatmap
    path = os.path.join(output_dir, 'metrics_heatmap.png')
    plot_metrics_heatmap(results_list, save_path=path, show_plot=show_plots)
    saved_plots['metrics_heatmap'] = path
    
    # Dashboard
    path = os.path.join(output_dir, 'evaluation_dashboard.png')
    plot_evaluation_dashboard(actuals, predictions_dict, results_list, dates,
                             save_path=path, show_plot=show_plots)
    saved_plots['dashboard'] = path
    
    # Individual model plots
    for model_name, predictions in predictions_dict.items():
        safe_name = model_name.lower().replace(' ', '_')
        
        # Predictions plot
        path = os.path.join(output_dir, f'{safe_name}_predictions.png')
        plot_predictions(actuals, predictions, dates, model_name,
                        save_path=path, show_plot=show_plots)
        saved_plots[f'{safe_name}_predictions'] = path
        
        # Error distribution
        path = os.path.join(output_dir, f'{safe_name}_errors.png')
        plot_error_distribution(actuals, predictions, model_name,
                               save_path=path, show_plot=show_plots)
        saved_plots[f'{safe_name}_errors'] = path
        
        # Residuals
        path = os.path.join(output_dir, f'{safe_name}_residuals.png')
        plot_residuals(actuals, predictions, dates, model_name,
                      save_path=path, show_plot=show_plots)
        saved_plots[f'{safe_name}_residuals'] = path
    
    if config.VERBOSE:
        print(f"\nAll plots saved to: {output_dir}/")
        print(f"Total plots generated: {len(saved_plots)}")
    
    return saved_plots


if __name__ == "__main__":
    print("=" * 60)
    print("Visualization Module - Example Usage")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start='2024-01-01', periods=n, freq='D')
    actuals = 100 + np.cumsum(np.random.randn(n) * 2)
    
    # Simulated model predictions
    predictions_dict = {
        'Naive': actuals + np.random.randn(n) * 3,
        'ARIMA': actuals + np.random.randn(n) * 4,
        'LSTM': actuals + np.random.randn(n) * 5,
    }
    
    # Simulated results
    from evaluation import evaluate_forecast
    results_list = [
        evaluate_forecast(actuals, predictions_dict['Naive'], 'Naive', verbose=False),
        evaluate_forecast(actuals, predictions_dict['ARIMA'], 'ARIMA', verbose=False),
        evaluate_forecast(actuals, predictions_dict['LSTM'], 'LSTM', verbose=False),
    ]
    
    print("\n--- Generating Plots ---")
    
    # Single model prediction
    print("\n1. Single Model Prediction Plot")
    plot_predictions(actuals, predictions_dict['ARIMA'], dates, 'ARIMA', show_plot=False)
    
    # Model comparison
    print("2. Model Comparison Plot")
    plot_model_comparison(actuals, predictions_dict, dates, show_plot=False)
    
    # Metrics comparison
    print("3. Metrics Comparison")
    plot_metrics_comparison(results_list, show_plot=False)
    
    # Dashboard
    print("4. Evaluation Dashboard")
    plot_evaluation_dashboard(actuals, predictions_dict, results_list, dates, show_plot=False)
    
    print("\n--- All plots generated successfully ---")
    print("Set show_plot=True to display plots interactively")
