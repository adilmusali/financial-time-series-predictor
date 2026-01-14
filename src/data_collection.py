import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Optional, Union


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def download_stock_data(
    ticker: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[Union[str, None]] = None,
    interval: Optional[str] = None,
    save_to_file: bool = True,
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    # Use config defaults if parameters not provided
    ticker = ticker or config.DEFAULT_TICKER
    start_date = start_date or config.START_DATE
    end_date = end_date if end_date is not None else config.END_DATE
    interval = interval or config.DATA_INTERVAL
    output_dir = output_dir or config.DATA_RAW_DIR
    
    if config.VERBOSE:
        print(f"Downloading data for ticker: {ticker}")
        print(f"Date range: {start_date} to {end_date or 'today'}")
        print(f"Interval: {interval}")
    
    try:
        # Create yfinance Ticker object
        stock = yf.Ticker(ticker)
        
        # Download historical data
        data = stock.history(
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True,
            prepost=False,
            repair=True
        )
        
        if data.empty:
            raise ValueError(f"No data retrieved for ticker {ticker}. "
                           f"Please check the ticker symbol and date range.")
        
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Rename columns to standard format (yfinance may return different names)
        # Standard columns: Open, High, Low, Close, Volume
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = [col for col in expected_columns if col in data.columns]
        
        if len(available_columns) < 4:  # At minimum need OHLC
            raise ValueError(f"Insufficient data columns. Expected OHLCV, got: {data.columns.tolist()}")
        
        # Select only OHLCV columns if they exist
        data = data[available_columns]
        
        # Remove any rows with all NaN values
        data = data.dropna(how='all')
        
        if config.VERBOSE:
            print(f"Successfully downloaded {len(data)} data points")
            print(f"Date range: {data.index.min()} to {data.index.max()}")
            print(f"Columns: {', '.join(data.columns.tolist())}")
        
        # Save to CSV if requested
        if save_to_file:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Create filename with ticker and date range
            start_str = start_date.replace('-', '')
            end_str = (end_date.replace('-', '') if end_date else 
                      datetime.now().strftime('%Y%m%d'))
            filename = f"{ticker}_{start_str}_{end_str}_{interval}.csv"
            filepath = os.path.join(output_dir, filename)
            
            # Save to CSV
            data.to_csv(filepath)
            
            if config.VERBOSE:
                print(f"Data saved to: {filepath}")
        
        return data
        
    except Exception as e:
        error_msg = f"Error downloading data for {ticker}: {str(e)}"
        if config.VERBOSE:
            print(error_msg)
        raise ValueError(error_msg) from e


def get_stock_info(ticker: Optional[str] = None) -> dict:
    ticker = ticker or config.DEFAULT_TICKER
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract key information
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
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        # If we can get info and it has a name, it's likely valid
        return bool(info.get('longName') or info.get('shortName'))
    except Exception:
        return False


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Data Collection Module - Example Usage")
    print("=" * 60)
    
    # Download default stock data
    data = download_stock_data()
    
    print("\nFirst few rows of downloaded data:")
    print(data.head())
    
    print("\nData summary:")
    print(data.describe())
    
    print("\nStock information:")
    get_stock_info()
