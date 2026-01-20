"""Main forecasting pipeline that orchestrates all components."""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from src.data_collection import download_stock_data
from src.data_cleaning import cleaned_data, data_statistics, data_validation
from src.exploratory_analysis import full_eda_report
from src.evaluation import (
    evaluate_forecast, compare_models, print_comparison_table,
    rank_models, generate_evaluation_report
)
from src.visualization import (
    plot_model_comparison, plot_metrics_comparison, plot_forecast,
    plot_multi_model_forecast, plot_train_test_split, plot_evaluation_dashboard,
    save_all_plots
)
from src.models.baseline import (
    train_test_split, NaiveModel, MovingAverageModel,
    evaluate_baseline, forecast_future
)
from src.models.arima_model import ARIMAModel, evaluate_arima, forecast_future_arima
from src.models.prophet_model import ProphetModel, evaluate_prophet, forecast_future_prophet
from src.models.lstm_model import LSTMModel, evaluate_lstm, forecast_future_lstm


class ForecastingPipeline:
    """End-to-end forecasting pipeline for data collection, training, and evaluation."""
    
    def __init__(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        prediction_horizon: Optional[int] = None,
        train_ratio: Optional[float] = None,
        verbose: bool = True
    ):
        self.ticker = ticker or config.DEFAULT_TICKER
        self.start_date = start_date or config.START_DATE
        self.end_date = end_date or config.END_DATE
        self.prediction_horizon = prediction_horizon or config.PREDICTION_HORIZON_DAYS
        self.train_ratio = train_ratio or config.TRAIN_TEST_SPLIT
        self.verbose = verbose
        self._original_verbose = config.VERBOSE
        
        self.raw_data = None
        self.cleaned_data = None
        self.train_data = None
        self.test_data = None
        self.models = {}
        self.results = {}
        self.forecasts = {}
        self.eda_results = None
        self.comparison_df = None
        self.best_model_name = None
    
    def _set_verbose(self) -> None:
        config.VERBOSE = self.verbose
    
    def _restore_verbose(self) -> None:
        config.VERBOSE = self._original_verbose
    
    def _print(self, message: str) -> None:
        if self.verbose:
            print(message)
    
    def collect_data(
        self,
        data_source: str = 'yfinance',
        filepath: Optional[str] = None,
        save_to_file: bool = True
    ) -> pd.DataFrame:
        """Step 1: Collect data from yfinance or CSV."""
        self._set_verbose()
        
        self._print("\n" + "=" * 70)
        self._print("STEP 1: DATA COLLECTION")
        self._print("=" * 70)
        
        self.raw_data = download_stock_data(
            ticker=self.ticker,
            start_date=self.start_date,
            end_date=self.end_date,
            data_source=data_source,
            filepath=filepath,
            save_to_file=save_to_file
        )
        
        self._restore_verbose()
        return self.raw_data
    
    def clean_data(
        self,
        data: Optional[pd.DataFrame] = None,
        missing_method: str = 'ffill',
        outlier_method: str = 'keep',
        save_to_file: bool = True
    ) -> pd.DataFrame:
        """Step 2: Clean and preprocess data."""
        self._set_verbose()
        
        self._print("\n" + "=" * 70)
        self._print("STEP 2: DATA CLEANING")
        self._print("=" * 70)
        
        data = data if data is not None else self.raw_data
        if data is None:
            raise ValueError("No data available. Call collect_data() first.")
        
        self.cleaned_data = cleaned_data(
            data,
            missing_method=missing_method,
            outlier_method=outlier_method,
            save_to_file=save_to_file
        )
        
        self._restore_verbose()
        return self.cleaned_data
    
    def split_data(
        self,
        data: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Step 3: Split data into train and test sets."""
        self._set_verbose()
        
        self._print("\n" + "=" * 70)
        self._print("STEP 3: TRAIN/TEST SPLIT")
        self._print("=" * 70)
        
        data = data if data is not None else self.cleaned_data
        if data is None:
            raise ValueError("No data available. Call clean_data() first.")
        
        self.train_data, self.test_data = train_test_split(data, self.train_ratio)
        
        self._restore_verbose()
        return self.train_data, self.test_data
    
    def run_eda(
        self,
        data: Optional[pd.DataFrame] = None,
        save_plots: bool = True
    ) -> Dict[str, Any]:
        """Step 4 (optional): Run exploratory data analysis."""
        self._set_verbose()
        
        self._print("\n" + "=" * 70)
        self._print("STEP 4: EXPLORATORY DATA ANALYSIS")
        self._print("=" * 70)
        
        data = data if data is not None else self.cleaned_data
        if data is None:
            raise ValueError("No data available. Call clean_data() first.")
        
        self.eda_results = full_eda_report(data, save_plots=save_plots)
        
        self._restore_verbose()
        return self.eda_results
    
    def train_baseline_models(
        self,
        train_data: Optional[pd.DataFrame] = None,
        column: str = 'Close'
    ) -> Dict[str, Any]:
        """Train baseline models (Naive, Moving Average)."""
        self._set_verbose()
        
        self._print("\n--- Training Baseline Models ---")
        
        train_data = train_data if train_data is not None else self.train_data
        if train_data is None:
            raise ValueError("No training data. Call split_data() first.")
        
        baseline_results = {}
        
        # Naive model
        naive = NaiveModel()
        naive.fit(train_data, column)
        naive_eval = evaluate_baseline(naive, self.test_data, column)
        self.models['Naive'] = naive
        baseline_results['Naive'] = naive_eval
        
        # Moving Average model
        ma = MovingAverageModel(window=config.MOVING_AVERAGE_WINDOW)
        ma.fit(train_data, column)
        ma_eval = evaluate_baseline(ma, self.test_data, column)
        self.models['Moving Average'] = ma
        baseline_results['Moving Average'] = ma_eval
        
        self.results.update(baseline_results)
        
        self._restore_verbose()
        return baseline_results
    
    def train_arima_model(
        self,
        train_data: Optional[pd.DataFrame] = None,
        column: str = 'Close',
        use_auto: bool = True,
        order: Optional[Tuple[int, int, int]] = None
    ) -> Dict[str, Any]:
        """Train ARIMA model with optional auto parameter selection."""
        self._set_verbose()
        
        self._print("\n--- Training ARIMA Model ---")
        
        train_data = train_data if train_data is not None else self.train_data
        if train_data is None:
            raise ValueError("No training data. Call split_data() first.")
        
        arima = ARIMAModel(order=order, use_auto=use_auto)
        arima.fit(train_data, column)
        
        arima_eval = evaluate_arima(arima, self.test_data, column)
        
        self.models['ARIMA'] = arima
        self.results['ARIMA'] = arima_eval
        
        self._restore_verbose()
        return {'ARIMA': arima_eval}
    
    def train_prophet_model(
        self,
        train_data: Optional[pd.DataFrame] = None,
        column: str = 'Close',
        **prophet_kwargs
    ) -> Dict[str, Any]:
        """Train Prophet model."""
        self._set_verbose()
        
        self._print("\n--- Training Prophet Model ---")
        
        train_data = train_data if train_data is not None else self.train_data
        if train_data is None:
            raise ValueError("No training data. Call split_data() first.")
        
        prophet = ProphetModel(**prophet_kwargs)
        prophet.fit(train_data, column)
        
        prophet_eval = evaluate_prophet(prophet, self.test_data, column)
        
        self.models['Prophet'] = prophet
        self.results['Prophet'] = prophet_eval
        
        self._restore_verbose()
        return {'Prophet': prophet_eval}
    
    def train_lstm_model(
        self,
        train_data: Optional[pd.DataFrame] = None,
        column: str = 'Close',
        **lstm_kwargs
    ) -> Dict[str, Any]:
        """Train LSTM model."""
        self._set_verbose()
        
        self._print("\n--- Training LSTM Model ---")
        
        train_data = train_data if train_data is not None else self.train_data
        if train_data is None:
            raise ValueError("No training data. Call split_data() first.")
        
        lstm = LSTMModel(**lstm_kwargs)
        lstm.fit(train_data, column)
        
        lstm_eval = evaluate_lstm(lstm, self.test_data, column)
        
        self.models['LSTM'] = lstm
        self.results['LSTM'] = lstm_eval
        
        self._restore_verbose()
        return {'LSTM': lstm_eval}
    
    def train_all_models(
        self,
        train_data: Optional[pd.DataFrame] = None,
        column: str = 'Close',
        include_lstm: bool = True,
        arima_use_auto: bool = True
    ) -> Dict[str, Any]:
        """Step 5: Train all models (Naive, MA, ARIMA, Prophet, LSTM)."""
        self._set_verbose()
        
        self._print("\n" + "=" * 70)
        self._print("STEP 5: MODEL TRAINING")
        self._print("=" * 70)
        
        train_data = train_data if train_data is not None else self.train_data
        if train_data is None:
            raise ValueError("No training data. Call split_data() first.")
        
        # Train baseline models
        self.train_baseline_models(train_data, column)
        
        # Train ARIMA
        self.train_arima_model(train_data, column, use_auto=arima_use_auto)
        
        # Train Prophet
        self.train_prophet_model(train_data, column)
        
        # Train LSTM (optional)
        if include_lstm:
            self.train_lstm_model(train_data, column)
        
        self._restore_verbose()
        return self.results
    
    def evaluate_models(self) -> pd.DataFrame:
        """Step 6: Compare and evaluate all trained models."""
        self._set_verbose()
        
        self._print("\n" + "=" * 70)
        self._print("STEP 6: MODEL EVALUATION")
        self._print("=" * 70)
        
        if not self.results:
            raise ValueError("No results available. Call train_all_models() first.")
        
        # Convert results to list format for comparison functions
        results_list = list(self.results.values())
        
        # Print comparison table
        print_comparison_table(results_list)
        
        # Get comparison DataFrame
        self.comparison_df = compare_models(results_list)
        
        # Find best model
        self.best_model_name = self.comparison_df.iloc[0]['Model']
        
        self._print(f"\nBest performing model: {self.best_model_name}")
        
        self._restore_verbose()
        return self.comparison_df
    
    def generate_forecasts(
        self,
        horizon: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """Step 7: Generate future forecasts from all models."""
        self._set_verbose()
        
        self._print("\n" + "=" * 70)
        self._print("STEP 7: GENERATING FORECASTS")
        self._print("=" * 70)
        
        horizon = horizon or self.prediction_horizon
        last_date = self.cleaned_data.index[-1]
        
        self.forecasts = {}
        
        for model_name, model in self.models.items():
            self._print(f"\nGenerating {horizon}-day forecast for {model_name}...")
            
            if model_name in ['Naive', 'Moving Average']:
                forecast_df = forecast_future(model, horizon, last_date)
                self.forecasts[model_name] = forecast_df
            elif model_name == 'ARIMA':
                forecast_df = forecast_future_arima(model, horizon, last_date)
                self.forecasts[model_name] = forecast_df
            elif model_name == 'Prophet':
                forecast_df = forecast_future_prophet(model, horizon, last_date)
                self.forecasts[model_name] = forecast_df
            elif model_name == 'LSTM':
                forecast_df = forecast_future_lstm(model, horizon, last_date)
                self.forecasts[model_name] = forecast_df
        
        self._restore_verbose()
        return self.forecasts
    
    def visualize_results(
        self,
        output_dir: Optional[str] = None,
        show_plots: bool = False
    ) -> Dict[str, str]:
        """Step 8: Generate and save visualizations."""
        self._set_verbose()
        
        self._print("\n" + "=" * 70)
        self._print("STEP 8: VISUALIZATION")
        self._print("=" * 70)
        
        output_dir = output_dir or config.PLOTS_DIR
        
        if not self.results:
            raise ValueError("No results to visualize. Call train_all_models() first.")
        
        # Prepare data for visualization
        actuals = self.test_data['Close'].values
        predictions_dict = {
            name: result['predictions']
            for name, result in self.results.items()
        }
        results_list = list(self.results.values())
        dates = self.test_data.index
        
        # Generate all plots
        saved_plots = save_all_plots(
            actuals=actuals,
            predictions_dict=predictions_dict,
            results_list=results_list,
            dates=dates,
            output_dir=output_dir,
            show_plots=show_plots
        )
        
        # Generate forecast comparison plot
        if self.forecasts:
            historical = self.cleaned_data['Close'].values
            forecasts_dict = {
                name: df['forecast'].values if 'forecast' in df.columns else df.values
                for name, df in self.forecasts.items()
            }
            
            forecast_path = os.path.join(output_dir, 'forecast_comparison.png')
            plot_multi_model_forecast(
                historical=historical,
                forecasts_dict=forecasts_dict,
                historical_dates=self.cleaned_data.index,
                forecast_dates=list(self.forecasts.values())[0].index if self.forecasts else None,
                save_path=forecast_path,
                show_plot=show_plots
            )
            saved_plots['forecast_comparison'] = forecast_path
        
        self._restore_verbose()
        return saved_plots
    
    def save_results(
        self,
        output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """Save forecasts and evaluation report to files."""
        self._set_verbose()
        
        output_dir = output_dir or config.DATA_PROCESSED_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save evaluation report
        if self.results:
            report_path = os.path.join(output_dir, f'evaluation_report_{timestamp}.txt')
            results_list = list(self.results.values())
            generate_evaluation_report(results_list, output_path=report_path)
            saved_files['evaluation_report'] = report_path
        
        # Save forecasts to CSV
        if self.forecasts:
            for model_name, forecast_df in self.forecasts.items():
                safe_name = model_name.lower().replace(' ', '_')
                forecast_path = os.path.join(
                    output_dir, 
                    f'{self.ticker}_{safe_name}_forecast_{timestamp}.csv'
                )
                forecast_df.to_csv(forecast_path)
                saved_files[f'{model_name}_forecast'] = forecast_path
            
            # Save combined forecasts
            combined_path = os.path.join(
                output_dir,
                f'{self.ticker}_all_forecasts_{timestamp}.csv'
            )
            combined_df = pd.DataFrame({
                name: df['forecast'] if 'forecast' in df.columns else df.iloc[:, 0]
                for name, df in self.forecasts.items()
            })
            combined_df.index = list(self.forecasts.values())[0].index
            combined_df.to_csv(combined_path)
            saved_files['combined_forecasts'] = combined_path
        
        # Save model comparison
        if self.comparison_df is not None:
            comparison_path = os.path.join(
                output_dir,
                f'{self.ticker}_model_comparison_{timestamp}.csv'
            )
            self.comparison_df.to_csv(comparison_path, index=False)
            saved_files['model_comparison'] = comparison_path
        
        self._print(f"\nResults saved to: {output_dir}/")
        for name, path in saved_files.items():
            self._print(f"  {name}: {os.path.basename(path)}")
        
        self._restore_verbose()
        return saved_files
    
    def run(
        self,
        run_eda: bool = False,
        include_lstm: bool = True,
        save_plots: bool = True,
        save_results: bool = True,
        show_plots: bool = False
    ) -> Dict[str, Any]:
        """Run the complete forecasting pipeline."""
        self._print("\n" + "=" * 70)
        self._print("FINANCIAL TIME SERIES FORECASTING PIPELINE")
        self._print("=" * 70)
        self._print(f"\nTicker: {self.ticker}")
        self._print(f"Date Range: {self.start_date} to {self.end_date or 'today'}")
        self._print(f"Prediction Horizon: {self.prediction_horizon} days")
        self._print(f"Train/Test Split: {self.train_ratio * 100:.0f}% / {(1-self.train_ratio) * 100:.0f}%")
        
        # Step 1: Collect data
        self.collect_data()
        
        # Step 2: Clean data
        self.clean_data()
        
        # Step 3: Split data
        self.split_data()
        
        # Step 4: EDA (optional)
        if run_eda:
            self.run_eda(save_plots=save_plots)
        
        # Step 5: Train models
        self.train_all_models(include_lstm=include_lstm)
        
        # Step 6: Evaluate models
        self.evaluate_models()
        
        # Step 7: Generate forecasts
        self.generate_forecasts()
        
        # Step 8: Visualize results
        if save_plots:
            self.visualize_results(show_plots=show_plots)
        
        # Save results
        if save_results:
            self.save_results()
        
        # Final summary
        self._print("\n" + "=" * 70)
        self._print("PIPELINE COMPLETE")
        self._print("=" * 70)
        self._print(f"\nBest Model: {self.best_model_name}")
        best_result = self.results.get(self.best_model_name, {})
        self._print(f"  MAE: ${best_result.get('mae', 0):.2f}")
        self._print(f"  RMSE: ${best_result.get('rmse', 0):.2f}")
        self._print(f"  MAPE: {best_result.get('mape', 0):.2f}%")
        
        return {
            'raw_data': self.raw_data,
            'cleaned_data': self.cleaned_data,
            'train_data': self.train_data,
            'test_data': self.test_data,
            'models': self.models,
            'results': self.results,
            'forecasts': self.forecasts,
            'comparison': self.comparison_df,
            'best_model': self.best_model_name,
            'eda_results': self.eda_results
        }


def run_forecast(
    ticker: str = None,
    start_date: str = None,
    end_date: str = None,
    prediction_horizon: int = None,
    run_eda: bool = False,
    include_lstm: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """Convenience function to run the complete forecasting pipeline."""
    pipeline = ForecastingPipeline(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        prediction_horizon=prediction_horizon,
        verbose=verbose
    )
    
    return pipeline.run(
        run_eda=run_eda,
        include_lstm=include_lstm
    )


if __name__ == "__main__":
    print("=" * 70)
    print("Forecasting Pipeline - Example Usage")
    print("=" * 70)
    
    # Run with default settings
    results = run_forecast(
        ticker='AAPL',
        run_eda=False,
        include_lstm=True,
        verbose=True
    )
    
    print("\n--- Pipeline Output ---")
    print(f"Models trained: {list(results['models'].keys())}")
    print(f"Best model: {results['best_model']}")
