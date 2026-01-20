#!/usr/bin/env python
"""
Financial Time Series Forecasting - Main Entry Point

Usage:
    python main.py                    # Default (AAPL)
    python main.py --ticker MSFT      # Microsoft
    python main.py --ticker GOOGL --eda
    python main.py --help
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.forecast import ForecastingPipeline, run_forecast


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Financial Time Series Forecasting Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --ticker MSFT
  python main.py --ticker AAPL --eda
  python main.py --ticker GOOGL --no-lstm
  python main.py --list-models
        """
    )
    
    parser.add_argument('--ticker', '-t', type=str, default=config.DEFAULT_TICKER,
                        help=f'Stock ticker (default: {config.DEFAULT_TICKER})')
    parser.add_argument('--start-date', '-s', type=str, default=config.START_DATE,
                        help=f'Start date YYYY-MM-DD (default: {config.START_DATE})')
    parser.add_argument('--end-date', '-e', type=str, default=None,
                        help='End date YYYY-MM-DD (default: today)')
    parser.add_argument('--csv', type=str, default=None,
                        help='Load from CSV instead of yfinance')
    parser.add_argument('--horizon', '-H', type=int, default=config.PREDICTION_HORIZON_DAYS,
                        help=f'Forecast horizon in days (default: {config.PREDICTION_HORIZON_DAYS})')
    parser.add_argument('--train-ratio', type=float, default=config.TRAIN_TEST_SPLIT,
                        help=f'Train/test split (default: {config.TRAIN_TEST_SPLIT})')
    parser.add_argument('--no-lstm', action='store_true', help='Skip LSTM (faster)')
    parser.add_argument('--no-arima-auto', action='store_true', help='Disable auto-ARIMA')
    parser.add_argument('--eda', action='store_true', help='Run EDA')
    parser.add_argument('--output-dir', '-o', type=str, default=config.DATA_PROCESSED_DIR,
                        help=f'Output directory (default: {config.DATA_PROCESSED_DIR})')
    parser.add_argument('--plots-dir', type=str, default=config.PLOTS_DIR,
                        help=f'Plots directory (default: {config.PLOTS_DIR})')
    parser.add_argument('--no-save', action='store_true', help='Do not save results')
    parser.add_argument('--show-plots', action='store_true', help='Display plots')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--list-models', action='store_true', help='List models and exit')
    parser.add_argument('--version', action='version', version='v1.0.0')
    return parser.parse_args()


def list_models():
    """Print available models."""
    print("\n" + "=" * 50)
    print("AVAILABLE MODELS")
    print("=" * 50)
    models = [
        ('Naive', 'Baseline', 'Last observed value'),
        ('Moving Average', 'Baseline', f'window={config.MOVING_AVERAGE_WINDOW}'),
        ('ARIMA', 'Statistical', f'order={config.ARIMA_ORDER}'),
        ('Prophet', 'Statistical', f'mode={config.PROPHET_SEASONALITY_MODE}'),
        ('LSTM', 'Deep Learning', f'units={config.LSTM_UNITS}'),
    ]
    for name, mtype, params in models:
        print(f"\n{name} ({mtype}): {params}")
    print("\n" + "-" * 50)
    print("Use --no-lstm to skip LSTM, --no-arima-auto for manual ARIMA")
    print()


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.list_models:
        list_models()
        return 0
    
    verbose = not args.quiet
    if args.plots_dir != config.PLOTS_DIR:
        config.PLOTS_DIR = args.plots_dir
    
    if verbose:
        print("\n" + "=" * 70)
        print("FINANCIAL TIME SERIES FORECASTING")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Ticker: {args.ticker}")
        print(f"  Date Range: {args.start_date} to {args.end_date or 'today'}")
        print(f"  Prediction Horizon: {args.horizon} days")
        print(f"  Train/Test Split: {args.train_ratio * 100:.0f}% / {(1-args.train_ratio) * 100:.0f}%")
        print(f"  Include EDA: {args.eda}")
        print(f"  Include LSTM: {not args.no_lstm}")
    
    try:
        pipeline = ForecastingPipeline(
            ticker=args.ticker, start_date=args.start_date, end_date=args.end_date,
            prediction_horizon=args.horizon, train_ratio=args.train_ratio, verbose=verbose
        )
        
        if args.csv:
            pipeline.collect_data(data_source='csv', filepath=args.csv)
        else:
            pipeline.collect_data(data_source='yfinance')
        
        pipeline.clean_data()
        pipeline.split_data()
        
        if args.eda:
            pipeline.run_eda(save_plots=not args.no_save)
        
        pipeline.train_all_models(include_lstm=not args.no_lstm, arima_use_auto=not args.no_arima_auto)
        pipeline.evaluate_models()
        pipeline.generate_forecasts()
        
        if not args.no_save:
            pipeline.visualize_results(output_dir=args.plots_dir, show_plots=args.show_plots)
            pipeline.save_results(output_dir=args.output_dir)
        
        if verbose:
            print("\n" + "=" * 70)
            print("FORECAST SUMMARY")
            print("=" * 70)
            print(f"\nBest Model: {pipeline.best_model_name}")
            best_result = pipeline.results.get(pipeline.best_model_name, {})
            print(f"  MAE: ${best_result.get('mae', 0):.2f}")
            print(f"  RMSE: ${best_result.get('rmse', 0):.2f}")
            print(f"  MAPE: {best_result.get('mape', 0):.2f}%")
            
            print(f"\n{args.horizon}-Day Forecast ({pipeline.best_model_name}):")
            if pipeline.best_model_name in pipeline.forecasts:
                forecast_df = pipeline.forecasts[pipeline.best_model_name]
                forecast_col = 'forecast' if 'forecast' in forecast_df.columns else forecast_df.columns[0]
                print(f"  First day: ${forecast_df[forecast_col].iloc[0]:.2f}")
                print(f"  Last day: ${forecast_df[forecast_col].iloc[-1]:.2f}")
            
            if not args.no_save:
                print(f"\nResults saved to:")
                print(f"  Data: {args.output_dir}/")
                print(f"  Plots: {args.plots_dir}/")
            
            print("\n" + "=" * 70)
            print("Forecasting complete!")
            print("=" * 70 + "\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
