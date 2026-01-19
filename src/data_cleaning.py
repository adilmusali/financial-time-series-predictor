"""Data cleaning module for time series preprocessing."""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def data_statistics(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Data statistics report.
    
    Args:
        data: DataFrame with OHLCV data and datetime index
        
    Returns:
        Dictionary containing data statistics
    """
    stats = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'columns': data.columns.tolist(),
        'dtypes': data.dtypes.to_dict(),
        'date_range': {
            'start': data.index.min(),
            'end': data.index.max(),
            'total_days': (data.index.max() - data.index.min()).days
        },
        'missing_values': data.isnull().sum().to_dict(),
        'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict(),
        'total_missing': data.isnull().sum().sum(),
        'duplicated_indices': data.index.duplicated().sum(),
        'basic_stats': data.describe().to_dict()
    }
    
    if config.VERBOSE:
        print("\n" + "=" * 60)
        print("DATA STATISTICS REPORT")
        print("=" * 60)
        print(f"\nShape: {stats['total_rows']} rows x {stats['total_columns']} columns")
        print(f"Columns: {', '.join(stats['columns'])}")
        print(f"\nDate Range:")
        print(f"  Start: {stats['date_range']['start']}")
        print(f"  End: {stats['date_range']['end']}")
        print(f"  Total days: {stats['date_range']['total_days']}")
        print(f"\nMissing Values:")
        for col, count in stats['missing_values'].items():
            pct = stats['missing_percentage'][col]
            print(f"  {col}: {count} ({pct:.2f}%)")
        print(f"\nTotal missing values: {stats['total_missing']}")
        print(f"Duplicated indices: {stats['duplicated_indices']}")
        print("\nData Types:")
        for col, dtype in stats['dtypes'].items():
            print(f"  {col}: {dtype}")
    
    return stats


def datetime_index_check(data: pd.DataFrame) -> pd.DataFrame:
    """
    Datetime index verification and correction.
    
    Args:
        data: DataFrame with potential datetime index
        
    Returns:
        DataFrame with proper datetime index
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        if config.VERBOSE:
            print("Converting index to DatetimeIndex...")
        data.index = pd.to_datetime(data.index)
    
    # Name the index if not named
    if data.index.name is None:
        data.index.name = 'Date'
    
    # Ensure timezone-naive for consistency
    if data.index.tz is not None:
        if config.VERBOSE:
            print(f"Removing timezone info (was {data.index.tz})...")
        data.index = data.index.tz_localize(None)
    
    return data


def duplicated_indices_handler(
    data: pd.DataFrame,
    strategy: str = 'keep_last'
) -> pd.DataFrame:
    """
    Duplicated datetime indices handler.
    
    Args:
        data: DataFrame with potential duplicated indices
        strategy: 'keep_first', 'keep_last', or 'average'
        
    Returns:
        DataFrame without duplicated indices
    """
    if not data.index.duplicated().any():
        if config.VERBOSE:
            print("No duplicated indices found.")
        return data
    
    num_duplicates = data.index.duplicated().sum()
    if config.VERBOSE:
        print(f"Found {num_duplicates} duplicated indices. Strategy: {strategy}")
    
    if strategy == 'keep_first':
        data = data[~data.index.duplicated(keep='first')]
    elif strategy == 'keep_last':
        data = data[~data.index.duplicated(keep='last')]
    elif strategy == 'average':
        # Group by index and average numeric columns
        data = data.groupby(data.index).mean()
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'keep_first', 'keep_last', or 'average'")
    
    return data


def missing_values_handler(
    data: pd.DataFrame,
    method: str = 'ffill',
    interpolation_method: str = 'linear',
    max_gap: Optional[int] = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Missing values handler for time series data.
    
    Args:
        data: DataFrame with potential missing values
        method: 'ffill' (forward fill), 'bfill' (backward fill), 
                'interpolate', 'drop', or 'ffill_bfill' (forward then backward)
        interpolation_method: Method for interpolation ('linear', 'time', 'polynomial', 'spline')
        max_gap: Maximum consecutive NaN values to fill (None = fill all)
        
    Returns:
        Tuple of (cleaned DataFrame, dictionary of filled counts per column)
    """
    # Count missing before
    missing_before = data.isnull().sum().to_dict()
    total_missing = data.isnull().sum().sum()
    
    if total_missing == 0:
        if config.VERBOSE:
            print("No missing values found.")
        return data, {col: 0 for col in data.columns}
    
    if config.VERBOSE:
        print(f"\nHandling {total_missing} missing values using method: {method}")
    
    data_cleaned = data.copy()
    
    if method == 'ffill':
        if max_gap:
            data_cleaned = data_cleaned.ffill(limit=max_gap)
        else:
            data_cleaned = data_cleaned.ffill()
        # Fill any remaining NaN at the start with backward fill
        data_cleaned = data_cleaned.bfill()
        
    elif method == 'bfill':
        if max_gap:
            data_cleaned = data_cleaned.bfill(limit=max_gap)
        else:
            data_cleaned = data_cleaned.bfill()
        # Fill any remaining NaN at the end with forward fill
        data_cleaned = data_cleaned.ffill()
        
    elif method == 'ffill_bfill':
        if max_gap:
            data_cleaned = data_cleaned.ffill(limit=max_gap).bfill(limit=max_gap)
        else:
            data_cleaned = data_cleaned.ffill().bfill()
            
    elif method == 'interpolate':
        if interpolation_method == 'time':
            data_cleaned = data_cleaned.interpolate(method='time', limit=max_gap)
        elif interpolation_method == 'polynomial':
            data_cleaned = data_cleaned.interpolate(method='polynomial', order=2, limit=max_gap)
        elif interpolation_method == 'spline':
            data_cleaned = data_cleaned.interpolate(method='spline', order=3, limit=max_gap)
        else:
            data_cleaned = data_cleaned.interpolate(method='linear', limit=max_gap)
        # Handle edge cases
        data_cleaned = data_cleaned.ffill().bfill()
        
    elif method == 'drop':
        rows_before = len(data_cleaned)
        data_cleaned = data_cleaned.dropna()
        rows_dropped = rows_before - len(data_cleaned)
        if config.VERBOSE:
            print(f"Dropped {rows_dropped} rows with missing values")
    else:
        raise ValueError(f"Unknown method: {method}. "
                        f"Use 'ffill', 'bfill', 'ffill_bfill', 'interpolate', or 'drop'")
    
    # Calculate filled counts
    filled_counts = {
        col: missing_before[col] - data_cleaned[col].isnull().sum() 
        for col in data.columns
    }
    
    if config.VERBOSE:
        print("Missing values handled:")
        for col, count in filled_counts.items():
            if count > 0:
                print(f"  {col}: {count} values filled")
    
    return data_cleaned, filled_counts


def outlier_detection(
    data: pd.DataFrame,
    column: str = 'Close',
    method: str = 'iqr',
    threshold: float = 1.5
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Outlier detection for specified column.
    
    Args:
        data: DataFrame with OHLCV data
        column: Column to check for outliers
        method: 'iqr' (Interquartile Range) or 'zscore' (Z-score)
        threshold: Multiplier for IQR method or z-score threshold
        
    Returns:
        Tuple of (boolean Series marking outliers, outlier statistics)
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    values = data[column]
    
    if method == 'iqr':
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (values < lower_bound) | (values > upper_bound)
        
        stats = {
            'method': 'IQR',
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'threshold': threshold,
            'num_outliers': outliers.sum(),
            'outlier_percentage': outliers.sum() / len(values) * 100
        }
        
    elif method == 'zscore':
        mean = values.mean()
        std = values.std()
        z_scores = np.abs((values - mean) / std)
        outliers = z_scores > threshold
        
        stats = {
            'method': 'Z-score',
            'mean': mean,
            'std': std,
            'threshold': threshold,
            'num_outliers': outliers.sum(),
            'outlier_percentage': outliers.sum() / len(values) * 100
        }
    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'")
    
    if config.VERBOSE:
        print(f"\nOutlier Detection ({stats['method']} method) for '{column}':")
        print(f"  Threshold: {threshold}")
        print(f"  Outliers found: {stats['num_outliers']} ({stats['outlier_percentage']:.2f}%)")
        if stats['num_outliers'] > 0:
            outlier_values = values[outliers]
            print(f"  Outlier range: {outlier_values.min():.2f} to {outlier_values.max():.2f}")
    
    return outliers, stats


def outlier_handler(
    data: pd.DataFrame,
    outlier_mask: pd.Series,
    column: str = 'Close',
    method: str = 'clip',
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None
) -> pd.DataFrame:
    """
    Outlier handler for detected outliers.
    
    Args:
        data: DataFrame with OHLCV data
        outlier_mask: Boolean Series marking outliers
        column: Column to handle outliers in
        method: 'clip' (cap at bounds), 'interpolate', 'remove', or 'keep'
        lower_bound: Lower bound for clipping (auto-calculated if None)
        upper_bound: Upper bound for clipping (auto-calculated if None)
        
    Returns:
        DataFrame with handled outliers
    """
    if not outlier_mask.any():
        if config.VERBOSE:
            print("No outliers to handle.")
        return data
    
    data_cleaned = data.copy()
    num_outliers = outlier_mask.sum()
    
    if method == 'keep':
        if config.VERBOSE:
            print(f"Keeping {num_outliers} outliers (no action taken)")
        return data_cleaned
    
    if config.VERBOSE:
        print(f"\nHandling {num_outliers} outliers using method: {method}")
    
    if method == 'clip':
        if lower_bound is None or upper_bound is None:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            if lower_bound is None:
                lower_bound = Q1 - 1.5 * IQR
            if upper_bound is None:
                upper_bound = Q3 + 1.5 * IQR
        
        data_cleaned[column] = data_cleaned[column].clip(lower=lower_bound, upper=upper_bound)
        if config.VERBOSE:
            print(f"  Clipped values to [{lower_bound:.2f}, {upper_bound:.2f}]")
            
    elif method == 'interpolate':
        data_cleaned.loc[outlier_mask, column] = np.nan
        data_cleaned[column] = data_cleaned[column].interpolate(method='linear')
        data_cleaned[column] = data_cleaned[column].ffill().bfill()
        if config.VERBOSE:
            print(f"  Interpolated {num_outliers} outlier values")
            
    elif method == 'remove':
        rows_before = len(data_cleaned)
        data_cleaned = data_cleaned[~outlier_mask]
        if config.VERBOSE:
            print(f"  Removed {rows_before - len(data_cleaned)} rows")
    else:
        raise ValueError(f"Unknown method: {method}. Use 'clip', 'interpolate', 'remove', or 'keep'")
    
    return data_cleaned


def data_continuity_report(
    data: pd.DataFrame,
    expected_frequency: str = 'B'
) -> Dict[str, Any]:
    """
    Data continuity report for time series gaps.
    
    Args:
        data: DataFrame with datetime index
        expected_frequency: Expected frequency ('D' for daily, 'B' for business days)
        
    Returns:
        Dictionary with gap information
    """
    # Create expected date range
    expected_dates = pd.date_range(
        start=data.index.min(),
        end=data.index.max(),
        freq=expected_frequency
    )
    
    # Find missing dates
    actual_dates = set(data.index.date)
    expected_dates_set = set(expected_dates.date)
    missing_dates = sorted(expected_dates_set - actual_dates)
    
    # Find gaps (consecutive missing dates)
    gaps = []
    if missing_dates:
        gap_start = missing_dates[0]
        gap_length = 1
        
        for i in range(1, len(missing_dates)):
            # Check if dates are consecutive
            delta = (missing_dates[i] - missing_dates[i-1]).days
            if delta <= 3:  # Allow for weekends
                gap_length += 1
            else:
                gaps.append({
                    'start': gap_start,
                    'end': missing_dates[i-1],
                    'length': gap_length
                })
                gap_start = missing_dates[i]
                gap_length = 1
        
        gaps.append({
            'start': gap_start,
            'end': missing_dates[-1],
            'length': gap_length
        })
    
    result = {
        'expected_frequency': expected_frequency,
        'expected_points': len(expected_dates),
        'actual_points': len(data),
        'missing_points': len(missing_dates),
        'missing_dates': missing_dates[:10] if len(missing_dates) > 10 else missing_dates,
        'total_missing_dates': len(missing_dates),
        'gaps': gaps[:5] if len(gaps) > 5 else gaps,
        'total_gaps': len(gaps)
    }
    
    if config.VERBOSE:
        print(f"\nData Continuity Check (expected freq: {expected_frequency}):")
        print(f"  Expected data points: {result['expected_points']}")
        print(f"  Actual data points: {result['actual_points']}")
        print(f"  Missing data points: {result['missing_points']}")
        if result['total_gaps'] > 0:
            print(f"  Number of gaps: {result['total_gaps']}")
            print("  First few gaps:")
            for gap in result['gaps'][:3]:
                print(f"    {gap['start']} to {gap['end']} ({gap['length']} days)")
    
    return result


def column_selection(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    target_column: str = 'Close'
) -> pd.DataFrame:
    """
    Column selection for analysis.
    
    Args:
        data: DataFrame with OHLCV data
        columns: List of columns to keep (None = keep all standard columns)
        target_column: Primary target column (must be included)
        
    Returns:
        DataFrame with selected columns
    """
    if columns is None:
        # Keep all available standard columns
        standard_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        columns = [col for col in standard_columns if col in data.columns]
    
    # Ensure target column is included
    if target_column not in columns:
        if target_column in data.columns:
            columns = [target_column] + columns
        else:
            raise ValueError(f"Target column '{target_column}' not found in data")
    
    # Filter to available columns
    available_columns = [col for col in columns if col in data.columns]
    
    if config.VERBOSE:
        print(f"\nSelected columns: {', '.join(available_columns)}")
    
    return data[available_columns]


def data_validation(
    data: pd.DataFrame,
    min_data_points: Optional[int] = None
) -> Tuple[bool, List[str]]:
    """
    Data validation for modeling readiness.
    
    Args:
        data: Cleaned DataFrame
        min_data_points: Minimum required data points (default from config)
        
    Returns:
        Tuple of (is_valid, list of issues)
    """
    min_data_points = min_data_points or config.MIN_DATA_POINTS
    issues = []
    
    # Check for minimum data points
    if len(data) < min_data_points:
        issues.append(f"Insufficient data points: {len(data)} < {min_data_points}")
    
    # Check for missing values
    missing = data.isnull().sum().sum()
    if missing > 0:
        issues.append(f"Contains {missing} missing values")
    
    # Check for datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        issues.append("Index is not DatetimeIndex")
    
    # Check for duplicated indices
    if data.index.duplicated().any():
        issues.append(f"Contains {data.index.duplicated().sum()} duplicated indices")
    
    # Check for sorted index
    if not data.index.is_monotonic_increasing:
        issues.append("Index is not sorted in ascending order")
    
    # Check for required columns
    if 'Close' not in data.columns:
        issues.append("Missing 'Close' column (required for prediction)")
    
    # Check for infinite values
    inf_count = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        issues.append(f"Contains {inf_count} infinite values")
    
    is_valid = len(issues) == 0
    
    if config.VERBOSE:
        print("\n" + "=" * 60)
        print("DATA VALIDATION REPORT")
        print("=" * 60)
        if is_valid:
            print("[OK] Data is valid and ready for modeling")
            print(f"  Total data points: {len(data)}")
            print(f"  Date range: {data.index.min()} to {data.index.max()}")
            print(f"  Columns: {', '.join(data.columns)}")
        else:
            print("[FAIL] Data validation failed:")
            for issue in issues:
                print(f"  - {issue}")
    
    return is_valid, issues


def cleaned_data(
    data: pd.DataFrame,
    missing_method: str = 'ffill',
    outlier_detection_flag: bool = True,
    outlier_method: str = 'keep',
    outlier_detection_type: str = 'iqr',
    outlier_threshold: float = 3.0,
    columns: Optional[List[str]] = None,
    save_to_file: bool = True,
    output_dir: Optional[str] = None,
    output_filename: Optional[str] = None
) -> pd.DataFrame:
    """
    Main data cleaning pipeline.
    
    Args:
        data: Raw DataFrame with OHLCV data
        missing_method: Method for handling missing values
        outlier_detection_flag: Whether to detect outliers
        outlier_method: Method for handling outliers
        outlier_detection_type: Method for detecting outliers ('iqr' or 'zscore')
        outlier_threshold: Threshold for outlier detection
        columns: Columns to keep (None = all standard columns)
        save_to_file: Whether to save cleaned data
        output_dir: Output directory (default: config.DATA_PROCESSED_DIR)
        output_filename: Output filename (auto-generated if None)
        
    Returns:
        Cleaned DataFrame ready for analysis and modeling
    """
    output_dir = output_dir or config.DATA_PROCESSED_DIR
    
    if config.VERBOSE:
        print("\n" + "=" * 60)
        print("STARTING DATA CLEANING PIPELINE")
        print("=" * 60)
    
    # Step 1: Analyze raw data
    if config.VERBOSE:
        print("\nStep 1: Analyzing raw data...")
    data_statistics(data)
    
    # Step 2: Ensure proper datetime index
    if config.VERBOSE:
        print("\nStep 2: Checking datetime index...")
    data = datetime_index_check(data)
    
    # Step 3: Handle duplicated indices
    if config.VERBOSE:
        print("\nStep 3: Handling duplicated indices...")
    data = duplicated_indices_handler(data)
    
    # Step 4: Sort by index
    if config.VERBOSE:
        print("\nStep 4: Sorting by date...")
    data = data.sort_index()
    
    # Step 5: Select columns
    if config.VERBOSE:
        print("\nStep 5: Selecting columns...")
    data = column_selection(data, columns)
    
    # Step 6: Handle missing values
    if config.VERBOSE:
        print("\nStep 6: Handling missing values...")
    data, filled_counts = missing_values_handler(data, method=missing_method)
    
    # Step 7: Detect and handle outliers
    if outlier_detection_flag:
        if config.VERBOSE:
            print("\nStep 7: Detecting outliers...")
        outlier_mask, outlier_stats = outlier_detection(
            data, 
            column='Close',
            method=outlier_detection_type,
            threshold=outlier_threshold
        )
        
        if outlier_stats['num_outliers'] > 0:
            data = outlier_handler(
                data, 
                outlier_mask, 
                column='Close',
                method=outlier_method
            )
    
    # Step 8: Check data continuity
    if config.VERBOSE:
        print("\nStep 8: Checking data continuity...")
    data_continuity_report(data)
    
    # Step 9: Validate final data
    if config.VERBOSE:
        print("\nStep 9: Validating cleaned data...")
    is_valid, issues = data_validation(data)
    
    if not is_valid:
        print("\nWarning: Data validation found issues that may affect modeling.")
    
    # Step 10: Save cleaned data
    if save_to_file:
        if config.VERBOSE:
            print("\nStep 10: Saving cleaned data...")
        os.makedirs(output_dir, exist_ok=True)
        
        if output_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"cleaned_data_{timestamp}.csv"
        
        filepath = os.path.join(output_dir, output_filename)
        data.to_csv(filepath)
        
        if config.VERBOSE:
            print(f"Cleaned data saved to: {filepath}")
    
    if config.VERBOSE:
        print("\n" + "=" * 60)
        print("DATA CLEANING COMPLETE")
        print("=" * 60)
        print(f"Final shape: {data.shape}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        print(f"Columns: {', '.join(data.columns)}")
    
    return data


def cleaned_data_loader(
    filepath: Optional[str] = None,
    directory: Optional[str] = None
) -> pd.DataFrame:
    """
    Cleaned data loader from file.
    
    Args:
        filepath: Path to specific file (overrides directory)
        directory: Directory to search for latest cleaned file
        
    Returns:
        DataFrame with cleaned data
    """
    if filepath is None:
        directory = directory or config.DATA_PROCESSED_DIR
        
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find the most recent cleaned data file
        csv_files = [f for f in os.listdir(directory) 
                    if f.endswith('.csv') and f.startswith('cleaned_')]
        
        if not csv_files:
            raise FileNotFoundError(f"No cleaned data files found in {directory}")
        
        # Get the most recent file
        csv_files.sort(reverse=True)
        filepath = os.path.join(directory, csv_files[0])
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if config.VERBOSE:
        print(f"Loading cleaned data from: {filepath}")
    
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    if config.VERBOSE:
        print(f"Loaded {len(data)} data points")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        print(f"Columns: {', '.join(data.columns)}")
    
    return data


if __name__ == "__main__":
    # Example usage
    from data_collection import download_stock_data
    
    print("=" * 60)
    print("Data Cleaning Module - Example Usage")
    print("=" * 60)
    
    # Download raw data
    print("\n--- Downloading raw data ---")
    raw_data = download_stock_data(save_to_file=True)
    
    # Clean the data
    print("\n--- Cleaning data ---")
    result = cleaned_data(
        raw_data,
        missing_method='ffill',
        outlier_detection_flag=True,
        outlier_method='keep',
        outlier_threshold=3.0,
        save_to_file=True
    )
    
    print("\n--- Cleaned Data Preview ---")
    print(result.head())
    print("\n--- Cleaned Data Statistics ---")
    print(result.describe())
