"""Exploratory Data Analysis module for time series analysis."""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict, Any, List, Union
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# =============================================================================
# Statistical Summary Functions
# =============================================================================

def summary_statistics(data: pd.DataFrame, column: str = 'Close') -> Dict[str, Any]:
    """Comprehensive statistical summary of the time series."""
    series = data[column]
    
    stats_dict = {
        'count': len(series),
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'variance': series.var(),
        'min': series.min(),
        'max': series.max(),
        'range': series.max() - series.min(),
        'skewness': series.skew(),
        'kurtosis': series.kurtosis(),
        'q1': series.quantile(0.25),
        'q3': series.quantile(0.75),
        'iqr': series.quantile(0.75) - series.quantile(0.25),
        'coefficient_of_variation': series.std() / series.mean() * 100,
    }
    
    # Date range info
    stats_dict['start_date'] = data.index.min()
    stats_dict['end_date'] = data.index.max()
    stats_dict['trading_days'] = len(data)
    stats_dict['calendar_days'] = (data.index.max() - data.index.min()).days
    
    if config.VERBOSE:
        print("\n" + "=" * 60)
        print(f"STATISTICAL SUMMARY - {column}")
        print("=" * 60)
        print(f"\nDate Range: {stats_dict['start_date'].strftime('%Y-%m-%d')} to {stats_dict['end_date'].strftime('%Y-%m-%d')}")
        print(f"Trading Days: {stats_dict['trading_days']} | Calendar Days: {stats_dict['calendar_days']}")
        print(f"\nCentral Tendency:")
        print(f"  Mean: ${stats_dict['mean']:.2f}")
        print(f"  Median: ${stats_dict['median']:.2f}")
        print(f"\nDispersion:")
        print(f"  Std Dev: ${stats_dict['std']:.2f}")
        print(f"  Range: ${stats_dict['range']:.2f} (${stats_dict['min']:.2f} - ${stats_dict['max']:.2f})")
        print(f"  IQR: ${stats_dict['iqr']:.2f}")
        print(f"  CV: {stats_dict['coefficient_of_variation']:.2f}%")
        print(f"\nDistribution Shape:")
        print(f"  Skewness: {stats_dict['skewness']:.4f}")
        print(f"  Kurtosis: {stats_dict['kurtosis']:.4f}")
    
    return stats_dict


def returns_analysis(data: pd.DataFrame, column: str = 'Close') -> Dict[str, Any]:
    """Analysis of returns (daily percentage changes)."""
    series = data[column]
    returns = series.pct_change().dropna()
    log_returns = np.log(series / series.shift(1)).dropna()
    
    stats_dict = {
        'mean_daily_return': returns.mean() * 100,
        'std_daily_return': returns.std() * 100,
        'annualized_return': returns.mean() * 252 * 100,
        'annualized_volatility': returns.std() * np.sqrt(252) * 100,
        'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
        'max_daily_gain': returns.max() * 100,
        'max_daily_loss': returns.min() * 100,
        'positive_days': (returns > 0).sum(),
        'negative_days': (returns < 0).sum(),
        'positive_ratio': (returns > 0).sum() / len(returns) * 100,
        'log_returns_mean': log_returns.mean() * 100,
        'log_returns_std': log_returns.std() * 100,
    }
    
    if config.VERBOSE:
        print("\n" + "=" * 60)
        print("RETURNS ANALYSIS")
        print("=" * 60)
        print(f"\nDaily Returns:")
        print(f"  Mean: {stats_dict['mean_daily_return']:.4f}%")
        print(f"  Std Dev: {stats_dict['std_daily_return']:.4f}%")
        print(f"\nAnnualized Metrics:")
        print(f"  Return: {stats_dict['annualized_return']:.2f}%")
        print(f"  Volatility: {stats_dict['annualized_volatility']:.2f}%")
        print(f"  Sharpe Ratio: {stats_dict['sharpe_ratio']:.4f}")
        print(f"\nExtreme Values:")
        print(f"  Max Daily Gain: {stats_dict['max_daily_gain']:.2f}%")
        print(f"  Max Daily Loss: {stats_dict['max_daily_loss']:.2f}%")
        print(f"\nWin/Loss:")
        print(f"  Positive Days: {stats_dict['positive_days']} ({stats_dict['positive_ratio']:.1f}%)")
        print(f"  Negative Days: {stats_dict['negative_days']}")
    
    return stats_dict


# =============================================================================
# Stationarity Tests
# =============================================================================

def stationarity_tests(data: pd.DataFrame, column: str = 'Close') -> Dict[str, Any]:
    """Stationarity tests (ADF and KPSS) for time series."""
    series = data[column].dropna()
    
    # ADF Test (null hypothesis: series has unit root / non-stationary)
    adf_result = adfuller(series, autolag='AIC')
    adf_dict = {
        'statistic': adf_result[0],
        'p_value': adf_result[1],
        'lags_used': adf_result[2],
        'n_observations': adf_result[3],
        'critical_values': adf_result[4],
        'is_stationary': adf_result[1] < 0.05
    }
    
    # KPSS Test (null hypothesis: series is stationary)
    kpss_result = kpss(series, regression='c', nlags='auto')
    kpss_dict = {
        'statistic': kpss_result[0],
        'p_value': kpss_result[1],
        'lags_used': kpss_result[2],
        'critical_values': kpss_result[3],
        'is_stationary': kpss_result[1] > 0.05
    }
    
    results = {
        'adf': adf_dict,
        'kpss': kpss_dict,
        'conclusion': 'stationary' if adf_dict['is_stationary'] and kpss_dict['is_stationary'] else 'non-stationary'
    }
    
    if config.VERBOSE:
        print("\n" + "=" * 60)
        print(f"STATIONARITY TESTS - {column}")
        print("=" * 60)
        print("\nAugmented Dickey-Fuller Test:")
        print(f"  H0: Series has unit root (non-stationary)")
        print(f"  Test Statistic: {adf_dict['statistic']:.4f}")
        print(f"  P-Value: {adf_dict['p_value']:.4f}")
        print(f"  Critical Values:")
        for key, val in adf_dict['critical_values'].items():
            print(f"    {key}: {val:.4f}")
        print(f"  Result: {'STATIONARY' if adf_dict['is_stationary'] else 'NON-STATIONARY'} (reject H0)" if adf_dict['is_stationary'] else f"  Result: NON-STATIONARY (fail to reject H0)")
        
        print("\nKPSS Test:")
        print(f"  H0: Series is stationary")
        print(f"  Test Statistic: {kpss_dict['statistic']:.4f}")
        print(f"  P-Value: {kpss_dict['p_value']:.4f}")
        print(f"  Result: {'STATIONARY' if kpss_dict['is_stationary'] else 'NON-STATIONARY'}")
        
        print(f"\nOverall Conclusion: {results['conclusion'].upper()}")
    
    return results


def differencing_analysis(
    data: pd.DataFrame, 
    column: str = 'Close',
    max_diff: int = 2
) -> Dict[str, Any]:
    """Determine optimal differencing order for stationarity."""
    series = data[column].dropna()
    results = {'original': stationarity_tests(pd.DataFrame({column: series}, index=series.index), column)}
    
    optimal_d = 0
    for d in range(1, max_diff + 1):
        diff_series = series.diff(d).dropna()
        diff_df = pd.DataFrame({column: diff_series}, index=diff_series.index)
        results[f'diff_{d}'] = stationarity_tests(diff_df, column)
        
        if results[f'diff_{d}']['conclusion'] == 'stationary' and optimal_d == 0:
            optimal_d = d
    
    results['optimal_d'] = optimal_d
    
    if config.VERBOSE:
        print(f"\nOptimal differencing order (d): {optimal_d}")
    
    return results


# =============================================================================
# Trend Analysis
# =============================================================================

def trend_detection(
    data: pd.DataFrame,
    column: str = 'Close',
    windows: List[int] = [20, 50, 200]
) -> Dict[str, Any]:
    """Trend detection using moving averages and linear regression."""
    series = data[column]
    
    # Calculate moving averages
    moving_averages = {}
    for window in windows:
        if len(series) >= window:
            moving_averages[f'SMA_{window}'] = series.rolling(window=window).mean()
    
    # Linear regression for overall trend
    x = np.arange(len(series))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
    
    # Trend direction based on slope
    if p_value < 0.05:
        if slope > 0:
            trend_direction = 'upward'
        else:
            trend_direction = 'downward'
    else:
        trend_direction = 'no significant trend'
    
    # Calculate trend strength
    trend_strength = abs(r_value)
    
    # Recent trend (last 20% of data)
    recent_start = int(len(series) * 0.8)
    recent_series = series.iloc[recent_start:]
    x_recent = np.arange(len(recent_series))
    recent_slope, _, recent_r, recent_p, _ = stats.linregress(x_recent, recent_series)
    
    if recent_p < 0.05:
        recent_trend = 'upward' if recent_slope > 0 else 'downward'
    else:
        recent_trend = 'sideways'
    
    results = {
        'moving_averages': moving_averages,
        'linear_regression': {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_error': std_err
        },
        'trend_direction': trend_direction,
        'trend_strength': trend_strength,
        'recent_trend': recent_trend,
        'daily_change_avg': series.diff().mean(),
        'total_change': series.iloc[-1] - series.iloc[0],
        'total_change_pct': (series.iloc[-1] / series.iloc[0] - 1) * 100
    }
    
    if config.VERBOSE:
        print("\n" + "=" * 60)
        print(f"TREND ANALYSIS - {column}")
        print("=" * 60)
        print(f"\nOverall Trend: {trend_direction.upper()}")
        print(f"  Slope: {slope:.4f} per day")
        print(f"  R-squared: {r_value**2:.4f}")
        print(f"  Trend Strength: {trend_strength:.4f}")
        print(f"\nRecent Trend (last 20%): {recent_trend.upper()}")
        print(f"\nPrice Change:")
        print(f"  Total Change: ${results['total_change']:.2f}")
        print(f"  Total Change %: {results['total_change_pct']:.2f}%")
        print(f"  Avg Daily Change: ${results['daily_change_avg']:.4f}")
    
    return results


def rolling_statistics(
    data: pd.DataFrame,
    column: str = 'Close',
    windows: List[int] = [7, 14, 30, 60]
) -> pd.DataFrame:
    """Calculate rolling statistics for multiple windows."""
    series = data[column]
    result = pd.DataFrame(index=data.index)
    result[column] = series
    
    for window in windows:
        if len(series) >= window:
            result[f'SMA_{window}'] = series.rolling(window=window).mean()
            result[f'STD_{window}'] = series.rolling(window=window).std()
            result[f'MIN_{window}'] = series.rolling(window=window).min()
            result[f'MAX_{window}'] = series.rolling(window=window).max()
    
    return result


# =============================================================================
# Seasonality Analysis
# =============================================================================

def seasonality_decomposition(
    data: pd.DataFrame,
    column: str = 'Close',
    period: Optional[int] = None,
    model: str = 'additive'
) -> Dict[str, Any]:
    """Seasonal decomposition of time series."""
    series = data[column].dropna()
    
    # Auto-detect period if not specified (try weekly = 5 trading days)
    if period is None:
        period = 5  # Weekly seasonality for trading days
    
    # Ensure we have enough data
    if len(series) < 2 * period:
        raise ValueError(f"Need at least {2 * period} observations for period {period}")
    
    # Perform decomposition
    decomposition = seasonal_decompose(series, model=model, period=period)
    
    # Calculate seasonal strength
    var_resid = decomposition.resid.dropna().var()
    var_seasonal = decomposition.seasonal.var()
    seasonal_strength = 1 - (var_resid / (var_resid + var_seasonal)) if (var_resid + var_seasonal) > 0 else 0
    
    # Calculate trend strength
    var_trend = decomposition.trend.dropna().var()
    trend_strength = 1 - (var_resid / (var_resid + var_trend)) if (var_resid + var_trend) > 0 else 0
    
    results = {
        'decomposition': decomposition,
        'trend': decomposition.trend,
        'seasonal': decomposition.seasonal,
        'residual': decomposition.resid,
        'period': period,
        'model': model,
        'seasonal_strength': seasonal_strength,
        'trend_strength': trend_strength,
        'has_strong_seasonality': seasonal_strength > 0.6,
        'has_strong_trend': trend_strength > 0.6
    }
    
    if config.VERBOSE:
        print("\n" + "=" * 60)
        print(f"SEASONAL DECOMPOSITION - {column}")
        print("=" * 60)
        print(f"\nModel: {model.upper()}")
        print(f"Period: {period}")
        print(f"\nComponent Strengths:")
        print(f"  Trend Strength: {trend_strength:.4f} ({'Strong' if results['has_strong_trend'] else 'Weak'})")
        print(f"  Seasonal Strength: {seasonal_strength:.4f} ({'Strong' if results['has_strong_seasonality'] else 'Weak'})")
    
    return results


def day_of_week_analysis(data: pd.DataFrame, column: str = 'Close') -> Dict[str, Any]:
    """Analyze patterns by day of week."""
    df = data.copy()
    df['returns'] = df[column].pct_change() * 100
    df['day_of_week'] = df.index.dayofweek
    df['day_name'] = df.index.day_name()
    
    # Group by day of week
    day_stats = df.groupby('day_name')['returns'].agg(['mean', 'std', 'count']).round(4)
    day_stats = day_stats.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
    
    # Find best and worst days
    best_day = day_stats['mean'].idxmax()
    worst_day = day_stats['mean'].idxmin()
    
    results = {
        'day_statistics': day_stats.to_dict(),
        'best_day': best_day,
        'best_day_return': day_stats.loc[best_day, 'mean'],
        'worst_day': worst_day,
        'worst_day_return': day_stats.loc[worst_day, 'mean'],
    }
    
    if config.VERBOSE:
        print("\n" + "=" * 60)
        print("DAY OF WEEK ANALYSIS")
        print("=" * 60)
        print("\nAverage Returns by Day:")
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            if day in day_stats.index:
                mean_ret = day_stats.loc[day, 'mean']
                std_ret = day_stats.loc[day, 'std']
                print(f"  {day}: {mean_ret:+.4f}% (std: {std_ret:.4f}%)")
        print(f"\nBest Day: {best_day} ({results['best_day_return']:+.4f}%)")
        print(f"Worst Day: {worst_day} ({results['worst_day_return']:+.4f}%)")
    
    return results


def monthly_analysis(data: pd.DataFrame, column: str = 'Close') -> Dict[str, Any]:
    """Analyze patterns by month."""
    df = data.copy()
    df['returns'] = df[column].pct_change() * 100
    df['month'] = df.index.month
    df['month_name'] = df.index.month_name()
    
    # Monthly returns (aggregate)
    monthly_returns = df.groupby(df.index.to_period('M'))[column].last().pct_change() * 100
    
    # Average by month
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    month_stats = df.groupby('month_name')['returns'].agg(['mean', 'std', 'count'])
    month_stats = month_stats.reindex(month_order).dropna()
    
    # Find best and worst months
    best_month = month_stats['mean'].idxmax()
    worst_month = month_stats['mean'].idxmin()
    
    results = {
        'month_statistics': month_stats.to_dict(),
        'best_month': best_month,
        'best_month_return': month_stats.loc[best_month, 'mean'],
        'worst_month': worst_month,
        'worst_month_return': month_stats.loc[worst_month, 'mean'],
        'monthly_returns': monthly_returns.to_dict()
    }
    
    if config.VERBOSE:
        print("\n" + "=" * 60)
        print("MONTHLY ANALYSIS")
        print("=" * 60)
        print("\nAverage Daily Returns by Month:")
        for month in month_order:
            if month in month_stats.index:
                mean_ret = month_stats.loc[month, 'mean']
                print(f"  {month}: {mean_ret:+.4f}%")
        print(f"\nBest Month: {best_month} ({results['best_month_return']:+.4f}%)")
        print(f"Worst Month: {worst_month} ({results['worst_month_return']:+.4f}%)")
    
    return results


# =============================================================================
# Autocorrelation Analysis
# =============================================================================

def autocorrelation_analysis(
    data: pd.DataFrame,
    column: str = 'Close',
    nlags: int = 40
) -> Dict[str, Any]:
    """Autocorrelation and partial autocorrelation analysis."""
    series = data[column].dropna()
    
    # Calculate ACF and PACF
    acf_values = acf(series, nlags=nlags, fft=True)
    pacf_values = pacf(series, nlags=nlags)
    
    # Find significant lags (using 95% confidence interval)
    confidence_interval = 1.96 / np.sqrt(len(series))
    significant_acf_lags = [i for i, v in enumerate(acf_values) if abs(v) > confidence_interval and i > 0]
    significant_pacf_lags = [i for i, v in enumerate(pacf_values) if abs(v) > confidence_interval and i > 0]
    
    # Also analyze returns
    returns = series.pct_change().dropna()
    returns_acf = acf(returns, nlags=nlags, fft=True)
    returns_pacf = pacf(returns, nlags=nlags)
    
    results = {
        'acf': acf_values,
        'pacf': pacf_values,
        'significant_acf_lags': significant_acf_lags[:10],
        'significant_pacf_lags': significant_pacf_lags[:10],
        'confidence_interval': confidence_interval,
        'returns_acf': returns_acf,
        'returns_pacf': returns_pacf,
        'nlags': nlags
    }
    
    if config.VERBOSE:
        print("\n" + "=" * 60)
        print(f"AUTOCORRELATION ANALYSIS - {column}")
        print("=" * 60)
        print(f"\n95% Confidence Interval: +/- {confidence_interval:.4f}")
        print(f"\nSignificant ACF Lags: {significant_acf_lags[:10]}")
        print(f"Significant PACF Lags: {significant_pacf_lags[:10]}")
        print(f"\nFirst 5 ACF values: {[f'{v:.4f}' for v in acf_values[1:6]]}")
        print(f"First 5 PACF values: {[f'{v:.4f}' for v in pacf_values[1:6]]}")
    
    return results


# =============================================================================
# Volume Analysis
# =============================================================================

def volume_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze trading volume patterns."""
    if 'Volume' not in data.columns:
        raise ValueError("Volume column not found in data")
    
    volume = data['Volume']
    close = data['Close']
    returns = close.pct_change()
    
    # Basic volume stats
    volume_stats = {
        'mean': volume.mean(),
        'median': volume.median(),
        'std': volume.std(),
        'min': volume.min(),
        'max': volume.max(),
    }
    
    # Volume-price correlation
    volume_return_corr = volume.corr(returns.abs())
    volume_price_corr = volume.corr(close)
    
    # Volume trends
    volume_sma_20 = volume.rolling(20).mean()
    recent_volume_vs_avg = volume.iloc[-20:].mean() / volume.mean()
    
    # High volume days (above 1.5x average)
    high_volume_threshold = volume.mean() * 1.5
    high_volume_days = (volume > high_volume_threshold).sum()
    
    results = {
        'statistics': volume_stats,
        'volume_return_correlation': volume_return_corr,
        'volume_price_correlation': volume_price_corr,
        'recent_vs_average_ratio': recent_volume_vs_avg,
        'high_volume_days': high_volume_days,
        'high_volume_pct': high_volume_days / len(volume) * 100,
        'volume_sma_20': volume_sma_20
    }
    
    if config.VERBOSE:
        print("\n" + "=" * 60)
        print("VOLUME ANALYSIS")
        print("=" * 60)
        print(f"\nVolume Statistics:")
        print(f"  Mean: {volume_stats['mean']:,.0f}")
        print(f"  Median: {volume_stats['median']:,.0f}")
        print(f"  Std Dev: {volume_stats['std']:,.0f}")
        print(f"\nCorrelations:")
        print(f"  Volume-Returns (abs): {volume_return_corr:.4f}")
        print(f"  Volume-Price: {volume_price_corr:.4f}")
        print(f"\nRecent Activity:")
        print(f"  Recent Volume vs Average: {recent_volume_vs_avg:.2f}x")
        print(f"  High Volume Days: {high_volume_days} ({results['high_volume_pct']:.1f}%)")
    
    return results


# =============================================================================
# Visualization Functions
# =============================================================================

def time_series_plot(
    data: pd.DataFrame,
    column: str = 'Close',
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Basic time series plot."""
    figsize = figsize or config.FIGURE_SIZE
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data.index, data[column], linewidth=1)
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{column} Price ($)')
    ax.set_title(title or f'{column} Price Over Time')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    
    return fig


def trend_plot(
    data: pd.DataFrame,
    column: str = 'Close',
    windows: List[int] = [20, 50, 200],
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Price with moving averages plot."""
    figsize = figsize or config.FIGURE_SIZE
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot price
    ax.plot(data.index, data[column], label=column, linewidth=1, alpha=0.7)
    
    # Plot moving averages
    colors = ['orange', 'green', 'red', 'purple']
    for i, window in enumerate(windows):
        if len(data) >= window:
            sma = data[column].rolling(window=window).mean()
            ax.plot(data.index, sma, label=f'SMA {window}', 
                   linewidth=1.5, color=colors[i % len(colors)])
    
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{column} Price ($)')
    ax.set_title(f'{column} Price with Moving Averages')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    
    return fig


def decomposition_plot(
    decomposition_result: Dict[str, Any],
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Seasonal decomposition plot."""
    figsize = figsize or (config.FIGURE_SIZE[0], config.FIGURE_SIZE[1] * 2)
    decomposition = decomposition_result['decomposition']
    
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    
    axes[0].plot(decomposition.observed)
    axes[0].set_ylabel('Observed')
    axes[0].set_title('Seasonal Decomposition')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(decomposition.trend)
    axes[1].set_ylabel('Trend')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(decomposition.seasonal)
    axes[2].set_ylabel('Seasonal')
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(decomposition.resid)
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Date')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    
    return fig


def acf_pacf_plot(
    data: pd.DataFrame,
    column: str = 'Close',
    nlags: int = 40,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """ACF and PACF plots."""
    figsize = figsize or (config.FIGURE_SIZE[0], config.FIGURE_SIZE[1] * 1.5)
    series = data[column].dropna()
    
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    plot_acf(series, lags=nlags, ax=axes[0], title=f'Autocorrelation Function (ACF) - {column}')
    plot_pacf(series, lags=nlags, ax=axes[1], title=f'Partial Autocorrelation Function (PACF) - {column}')
    
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    
    return fig


def distribution_plot(
    data: pd.DataFrame,
    column: str = 'Close',
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Distribution and returns histogram plot."""
    figsize = figsize or (config.FIGURE_SIZE[0], config.FIGURE_SIZE[1] * 1.5)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Price distribution
    axes[0, 0].hist(data[column], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel(f'{column} Price ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'{column} Price Distribution')
    axes[0, 0].axvline(data[column].mean(), color='red', linestyle='--', label=f'Mean: ${data[column].mean():.2f}')
    axes[0, 0].legend()
    
    # Returns distribution
    returns = data[column].pct_change().dropna() * 100
    axes[0, 1].hist(returns, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Daily Returns (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Daily Returns Distribution')
    axes[0, 1].axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.2f}%')
    axes[0, 1].legend()
    
    # Q-Q plot for returns
    stats.probplot(returns, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Returns vs Normal)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1, 1].boxplot([data[column].values, returns.values], labels=[column, 'Returns (%)'])
    axes[1, 1].set_title('Box Plots')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    
    return fig


def volume_price_plot(
    data: pd.DataFrame,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Volume and price combined plot."""
    figsize = figsize or (config.FIGURE_SIZE[0], config.FIGURE_SIZE[1] * 1.5)
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, 
                             gridspec_kw={'height_ratios': [2, 1]})
    
    # Price plot
    axes[0].plot(data.index, data['Close'], linewidth=1)
    axes[0].set_ylabel('Close Price ($)')
    axes[0].set_title('Price and Volume')
    axes[0].grid(True, alpha=0.3)
    
    # Volume plot
    colors = ['green' if data['Close'].iloc[i] >= data['Close'].iloc[i-1] else 'red' 
              for i in range(1, len(data))]
    colors = ['gray'] + colors
    axes[1].bar(data.index, data['Volume'], color=colors, alpha=0.7, width=1)
    axes[1].set_ylabel('Volume')
    axes[1].set_xlabel('Date')
    axes[1].grid(True, alpha=0.3)
    
    # Add volume SMA
    volume_sma = data['Volume'].rolling(20).mean()
    axes[1].plot(data.index, volume_sma, color='blue', linewidth=1.5, label='20-day SMA')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    
    return fig


# =============================================================================
# Comprehensive EDA Report
# =============================================================================

def full_eda_report(
    data: pd.DataFrame,
    column: str = 'Close',
    save_plots: bool = True,
    plots_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Generate comprehensive EDA report."""
    plots_dir = plots_dir or config.PLOTS_DIR
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EXPLORATORY DATA ANALYSIS REPORT")
    print("=" * 80)
    
    results = {}
    
    # 1. Statistical Summary
    print("\n[1/8] Computing statistical summary...")
    results['summary_statistics'] = summary_statistics(data, column)
    
    # 2. Returns Analysis
    print("[2/8] Analyzing returns...")
    results['returns_analysis'] = returns_analysis(data, column)
    
    # 3. Stationarity Tests
    print("[3/8] Running stationarity tests...")
    results['stationarity'] = stationarity_tests(data, column)
    
    # 4. Trend Analysis
    print("[4/8] Detecting trends...")
    results['trend'] = trend_detection(data, column)
    
    # 5. Seasonality Analysis
    print("[5/8] Analyzing seasonality...")
    try:
        results['seasonality'] = seasonality_decomposition(data, column)
        results['day_of_week'] = day_of_week_analysis(data, column)
        results['monthly'] = monthly_analysis(data, column)
    except Exception as e:
        print(f"  Warning: Seasonality analysis failed: {e}")
        results['seasonality'] = None
    
    # 6. Autocorrelation
    print("[6/8] Computing autocorrelation...")
    results['autocorrelation'] = autocorrelation_analysis(data, column)
    
    # 7. Volume Analysis
    print("[7/8] Analyzing volume...")
    if 'Volume' in data.columns:
        results['volume'] = volume_analysis(data)
    else:
        results['volume'] = None
        print("  Warning: Volume column not found")
    
    # 8. Generate Plots
    if save_plots:
        print("[8/8] Generating plots...")
        os.makedirs(plots_dir, exist_ok=True)
        
        time_series_plot(data, column, save_path=os.path.join(plots_dir, 'time_series.png'))
        trend_plot(data, column, save_path=os.path.join(plots_dir, 'trend_analysis.png'))
        acf_pacf_plot(data, column, save_path=os.path.join(plots_dir, 'acf_pacf.png'))
        distribution_plot(data, column, save_path=os.path.join(plots_dir, 'distribution.png'))
        
        if results['seasonality']:
            decomposition_plot(results['seasonality'], 
                             save_path=os.path.join(plots_dir, 'decomposition.png'))
        
        if 'Volume' in data.columns:
            volume_price_plot(data, save_path=os.path.join(plots_dir, 'volume_price.png'))
        
        print(f"  Plots saved to: {plots_dir}/")
    
    print("\n" + "=" * 80)
    print("EDA REPORT COMPLETE")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    from data_cleaning import cleaned_data_loader
    
    print("=" * 60)
    print("Exploratory Analysis Module - Example Usage")
    print("=" * 60)
    
    # Load cleaned data
    try:
        data = cleaned_data_loader()
    except FileNotFoundError:
        from data_collection import download_stock_data
        from data_cleaning import cleaned_data
        
        print("\nNo cleaned data found. Downloading and cleaning...")
        raw_data = download_stock_data()
        data = cleaned_data(raw_data)
    
    # Run full EDA
    results = full_eda_report(data, save_plots=True)
