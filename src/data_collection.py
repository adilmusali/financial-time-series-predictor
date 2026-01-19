"""Data collection module - supports yfinance and CSV sources."""

import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Optional, Union, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _standardize_data(data: pd.DataFrame, source_name: str = '') -> pd.DataFrame:
    """Standardize data format: datetime index and OHLCV columns."""
    if data.empty:
        raise ValueError(f"No data retrieved from {source_name}. "
                       f"Please check the data source and parameters.")
    
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Standardize column names (case-insensitive)
    column_mapping = {}
    for col in data.columns:
        col_lower = col.lower()
        if 'open' in col_lower:
            column_mapping[col] = 'Open'
        elif 'high' in col_lower:
            column_mapping[col] = 'High'
        elif 'low' in col_lower:
            column_mapping[col] = 'Low'
        elif 'close' in col_lower:
            column_mapping[col] = 'Close'
        elif 'volume' in col_lower:
            column_mapping[col] = 'Volume'
    
    if column_mapping:
        data = data.rename(columns=column_mapping)
    
    expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    available_columns = [col for col in expected_columns if col in data.columns]
    
    if len(available_columns) < 4:
        raise ValueError(f"Insufficient data columns from {source_name}. "
                        f"Expected OHLCV, got: {data.columns.tolist()}")
    
    data = data[available_columns]
    data = data.dropna(how='all')
    data = data.sort_index()
    
    return data


def _download_from_yfinance(
    ticker: str,
    start_date: str,
    end_date: Optional[str],
    interval: str
) -> pd.DataFrame:
    """Download data from yfinance."""
    if config.VERBOSE:
        print(f"Downloading from yfinance for ticker: {ticker}")
    
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True,
            prepost=False,
            repair=True
        )
        
        return _standardize_data(data, 'yfinance')
        
    except Exception as e:
        raise ValueError(f"Error downloading from yfinance for {ticker}: {str(e)}") from e


def _load_from_csv(
    filepath: str,
    date_column: Optional[str] = None,
    index_column: Optional[int] = None
) -> pd.DataFrame:
    """Load data from CSV file."""
    if config.VERBOSE:
        print(f"Loading data from CSV: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    try:
        if date_column:
            data = pd.read_csv(filepath, parse_dates=[date_column], index_col=date_column)
        elif index_column is not None:
            data = pd.read_csv(filepath, index_col=index_column, parse_dates=True)
        else:
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        return _standardize_data(data, 'CSV')
        
    except Exception as e:
        raise ValueError(f"Error loading CSV file {filepath}: {str(e)}") from e


def download_stock_data(
    ticker: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[Union[str, None]] = None,
    interval: Optional[str] = None,
    data_source: Optional[str] = None,
    filepath: Optional[str] = None,
    save_to_file: bool = True,
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Download or load stock data from multiple sources.
    
    Args:
        ticker: Stock ticker (required for 'yfinance')
        start_date: Start date 'YYYY-MM-DD' (for 'yfinance')
        end_date: End date 'YYYY-MM-DD' (for 'yfinance', None = today)
        interval: Data interval '1d', '1wk', '1mo' (for 'yfinance')
        data_source: 'yfinance' or 'csv' (default: config or 'yfinance')
        filepath: CSV file path (required for 'csv')
        save_to_file: Save to CSV
        output_dir: Output directory (default: config.DATA_RAW_DIR)
    
    Returns:
        DataFrame with OHLCV data and datetime index
    """
    data_source = (data_source or getattr(config, 'DEFAULT_DATA_SOURCE', 'yfinance')).lower()
    output_dir = output_dir or config.DATA_RAW_DIR
    if data_source == 'yfinance':
        ticker = ticker or config.DEFAULT_TICKER
        start_date = start_date or config.START_DATE
        end_date = end_date if end_date is not None else config.END_DATE
        interval = interval or config.DATA_INTERVAL
        
        if config.VERBOSE:
            print(f"Data source: yfinance")
            print(f"Ticker: {ticker}")
            print(f"Date range: {start_date} to {end_date or 'today'}")
            print(f"Interval: {interval}")
        
        data = _download_from_yfinance(ticker, start_date, end_date, interval)
        source_identifier = ticker
        
    elif data_source == 'csv':
        if filepath is None:
            raise ValueError("filepath parameter is required when data_source='csv'")
        
        if config.VERBOSE:
            print(f"Data source: CSV file")
            print(f"Filepath: {filepath}")
        
        data = _load_from_csv(filepath)
        source_identifier = os.path.splitext(os.path.basename(filepath))[0]
        
    else:
        raise ValueError(f"Unsupported data source: {data_source}. "
                        f"Supported sources: 'yfinance', 'csv'")
    
    if config.VERBOSE:
        print(f"Successfully loaded {len(data)} data points")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        print(f"Columns: {', '.join(data.columns.tolist())}")
    
    if save_to_file:
        os.makedirs(output_dir, exist_ok=True)
        
        if data_source == 'yfinance':
            start_str = start_date.replace('-', '')
            end_str = (end_date.replace('-', '') if end_date else 
                      datetime.now().strftime('%Y%m%d'))
            filename = f"{source_identifier}_{start_str}_{end_str}_{interval}.csv"
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{source_identifier}_loaded_{timestamp}.csv"
        
        filepath = os.path.join(output_dir, filename)
        data.to_csv(filepath)
        
        if config.VERBOSE:
            print(f"Data saved to: {filepath}")
    
    return data


def get_stock_info(ticker: Optional[str] = None) -> dict:
    """Get stock information from yfinance."""
    ticker = ticker or config.DEFAULT_TICKER
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        key_info = {
            'symbol': ticker,
            'name': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'currency': info.get('currency', 'N/A'),
            'exchange': info.get('exchange', 'N/A')
        }
        
        if config.VERBOSE:
            print(f"\nStock Information for {ticker}:")
            for key, value in key_info.items():
                print(f"  {key.capitalize()}: {value}")
        
        return key_info
        
    except Exception as e:
        if config.VERBOSE:
            print(f"Error retrieving info for {ticker}: {str(e)}")
        return {'symbol': ticker, 'error': str(e)}


def validate_ticker(ticker: str) -> bool:
    """Validate ticker symbol (yfinance only)."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return bool(info.get('longName') or info.get('shortName'))
    except Exception:
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Data Collection Module - Example Usage")
    print("=" * 60)
    
    print("\n--- Example 1: Download from yfinance ---")
    data = download_stock_data()
    print("\nFirst few rows:")
    print(data.head())
    
    # Example 2: Load from CSV
    # csv_data = download_stock_data(
    #     data_source='csv',
    #     filepath='data/raw/your_file.csv',
    #     save_to_file=False
    # )
    
    print("\n--- Data Summary ---")
    print(data.describe())
    
    print("\n--- Stock Information ---")
    get_stock_info()
