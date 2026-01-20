"""Prophet model implementation for time series forecasting."""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

# Suppress Prophet and cmdstanpy logging
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import logging
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

from prophet import Prophet

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

from src.models.baseline import train_test_split


def prepare_prophet_data(
    data: pd.DataFrame,
    column: str = 'Close'
) -> pd.DataFrame:
    """
    Prepare data in Prophet format (ds, y).
    """
    prophet_df = pd.DataFrame({
        'ds': data.index,
        'y': data[column].values
    })
    
    # Ensure ds is datetime
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    
    return prophet_df


def create_future_dataframe(
    last_date: pd.Timestamp,
    horizon: int,
    freq: str = 'B'
) -> pd.DataFrame:
    """
    Create future dates dataframe for Prophet prediction.
    """
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon
    )
    
    return pd.DataFrame({'ds': future_dates})


class ProphetModel:
    """Facebook Prophet model for time series forecasting with trend and seasonality."""
    
    def __init__(
        self,
        changepoint_prior_scale: Optional[float] = None,
        seasonality_mode: Optional[str] = None,
        yearly_seasonality: Optional[bool] = None,
        weekly_seasonality: Optional[bool] = None,
        daily_seasonality: Optional[bool] = None,
        interval_width: Optional[float] = None,
        growth: str = 'linear',
        n_changepoints: int = 25,
        changepoint_range: float = 0.8,
        holidays: Optional[pd.DataFrame] = None,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0
    ):
        """
        Initialize Prophet model.
        """
        # Use config values if not specified
        self.changepoint_prior_scale = changepoint_prior_scale or config.PROPHET_CHANGEPOINT_PRIOR_SCALE
        self.seasonality_mode = seasonality_mode or config.PROPHET_SEASONALITY_MODE
        self.yearly_seasonality = yearly_seasonality if yearly_seasonality is not None else config.PROPHET_YEARLY_SEASONALITY
        self.weekly_seasonality = weekly_seasonality if weekly_seasonality is not None else config.PROPHET_WEEKLY_SEASONALITY
        self.daily_seasonality = daily_seasonality if daily_seasonality is not None else config.PROPHET_DAILY_SEASONALITY
        self.interval_width = interval_width or config.PROPHET_INTERVAL_WIDTH
        self.growth = growth
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.holidays = holidays
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        
        self.model = None
        self.is_fitted = False
        self.train_data = None
        self.prophet_data = None
        self.column = None
        self.forecast_df = None
        
        self._name = "Prophet"
    
    @property
    def name(self) -> str:
        """Return model name."""
        return self._name
    
    def _create_model(self) -> Prophet:
        """Create Prophet model instance with configured parameters."""
        model = Prophet(
            growth=self.growth,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            interval_width=self.interval_width,
            n_changepoints=self.n_changepoints,
            changepoint_range=self.changepoint_range,
            holidays=self.holidays,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale
        )
        
        return model
    
    def fit(
        self,
        train_data: pd.DataFrame,
        column: str = 'Close'
    ) -> 'ProphetModel':
        """
        Fit Prophet model to training data.
        """
        self.train_data = train_data.copy()
        self.column = column
        
        if config.VERBOSE:
            print(f"\n--- Fitting Prophet Model ---")
            print(f"Training samples: {len(train_data)}")
            print(f"Date range: {train_data.index.min().strftime('%Y-%m-%d')} to {train_data.index.max().strftime('%Y-%m-%d')}")
            print(f"Seasonality mode: {self.seasonality_mode}")
            print(f"Yearly seasonality: {self.yearly_seasonality}")
            print(f"Weekly seasonality: {self.weekly_seasonality}")
        
        # Prepare data for Prophet
        self.prophet_data = prepare_prophet_data(train_data, column)
        
        # Create and fit model
        self.model = self._create_model()
        
        # Suppress stdout during fitting
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.prophet_data)
        
        self.is_fitted = True
        
        if config.VERBOSE:
            print("Model fitted successfully!")
        
        return self
    
    def predict(self, horizon: int) -> pd.Series:
        """
        Generate out-of-sample forecasts.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Create future dataframe
        future = create_future_dataframe(
            self.train_data.index[-1],
            horizon
        )
        
        # Get predictions
        forecast = self.model.predict(future)
        self.forecast_df = forecast
        
        return pd.Series(forecast['yhat'].values)
    
    def predict_with_confidence(
        self,
        horizon: int
    ) -> pd.DataFrame:
        """
        Generate forecasts with confidence intervals.           
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Create future dataframe
        future = create_future_dataframe(
            self.train_data.index[-1],
            horizon
        )
        
        # Get predictions
        forecast = self.model.predict(future)
        self.forecast_df = forecast
        
        # Create result DataFrame
        result = pd.DataFrame({
            'forecast': forecast['yhat'].values,
            'lower': forecast['yhat_lower'].values,
            'upper': forecast['yhat_upper'].values
        })
        
        return result
    
    def predict_in_sample(self) -> pd.DataFrame:
        """
        Generate in-sample predictions (fitted values).
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Predict on training data
        forecast = self.model.predict(self.prophet_data)
        
        return forecast
    
    def get_components(self) -> pd.DataFrame:
        """
        Get decomposed components (trend, seasonality).
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get in-sample predictions with components
        forecast = self.model.predict(self.prophet_data)
        
        components = pd.DataFrame({
            'ds': forecast['ds'],
            'trend': forecast['trend'],
            'yhat': forecast['yhat']
        })
        
        # Add yearly seasonality if present
        if 'yearly' in forecast.columns:
            components['yearly'] = forecast['yearly']
        
        # Add weekly seasonality if present
        if 'weekly' in forecast.columns:
            components['weekly'] = forecast['weekly']
        
        # Add holidays if present
        if 'holidays' in forecast.columns:
            components['holidays'] = forecast['holidays']
        
        return components
    
    def get_changepoints(self) -> pd.DataFrame:
        """
        Get detected changepoints in the trend.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get changepoints
        changepoints = self.model.changepoints
        
        if len(changepoints) == 0:
            return pd.DataFrame(columns=['ds', 'delta'])
        
        # Get changepoint magnitudes
        deltas = self.model.params['delta'].mean(axis=0)
        
        return pd.DataFrame({
            'ds': changepoints,
            'delta': deltas[:len(changepoints)]
        })
    
    def plot_components(self, figsize: Tuple[int, int] = (12, 10)):
        """
        Plot model components (trend, seasonality).
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        import matplotlib.pyplot as plt
        
        # Get forecast for plotting
        forecast = self.model.predict(self.prophet_data)
        
        # Use Prophet's built-in plotting
        fig = self.model.plot_components(forecast)
        fig.set_size_inches(figsize)
        plt.tight_layout()
        
        return fig
    
    def plot_forecast(
        self,
        forecast_df: Optional[pd.DataFrame] = None,
        figsize: Tuple[int, int] = (14, 6)
    ):
        """
        Plot forecast with uncertainty intervals.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        import matplotlib.pyplot as plt
        
        if forecast_df is None:
            # Predict on full data + future
            future = self.model.make_future_dataframe(periods=config.PREDICTION_HORIZON_DAYS)
            forecast_df = self.model.predict(future)
        
        fig = self.model.plot(forecast_df)
        fig.set_size_inches(figsize)
        plt.title(f'{self.name} Forecast', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def cross_validate(
        self,
        initial: str = '730 days',
        period: str = '180 days',
        horizon: str = '30 days'
    ) -> pd.DataFrame:
        """
        Perform time series cross-validation.
        
        Parameters
        ----------
        initial : str
            Initial training period
        period : str
            Period between cutoff dates
        horizon : str
            Forecast horizon
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        from prophet.diagnostics import cross_validation
        
        cv_results = cross_validation(
            self.model,
            initial=initial,
            period=period,
            horizon=horizon
        )
        
        return cv_results
    
    def performance_metrics(
        self,
        cv_results: Optional[pd.DataFrame] = None,
        rolling_window: float = 0.1
    ) -> pd.DataFrame:
        """
        Calculate performance metrics from cross-validation.
        """
        from prophet.diagnostics import performance_metrics
        
        if cv_results is None:
            cv_results = self.cross_validate()
        
        metrics = performance_metrics(cv_results, rolling_window=rolling_window)
        
        return metrics
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'growth': self.growth,
            'changepoint_prior_scale': self.changepoint_prior_scale,
            'seasonality_mode': self.seasonality_mode,
            'yearly_seasonality': self.yearly_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'daily_seasonality': self.daily_seasonality,
            'interval_width': self.interval_width,
            'n_changepoints': self.n_changepoints,
            'changepoint_range': self.changepoint_range,
            'seasonality_prior_scale': self.seasonality_prior_scale,
            'holidays_prior_scale': self.holidays_prior_scale
        }


def evaluate_prophet(
    model: ProphetModel,
    test_data: pd.DataFrame,
    column: str = 'Close'
) -> Dict[str, Any]:
    """
    Evaluate Prophet model on test data.
    
    Parameters
    ----------
    model : ProphetModel
        Fitted Prophet model
    test_data : pd.DataFrame
        Test data for evaluation
    column : str
        Column name to evaluate
    """
    actuals = test_data[column].values
    
    # Get predictions for test period
    predictions_df = model.predict_with_confidence(len(test_data))
    predictions = predictions_df['forecast'].values
    
    # Calculate metrics
    errors = actuals - predictions
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    mape = np.mean(np.abs(errors / actuals)) * 100
    
    results = {
        'model': model.name,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'predictions': predictions,
        'lower_bound': predictions_df['lower'].values,
        'upper_bound': predictions_df['upper'].values,
        'actuals': actuals,
        'errors': errors
    }
    
    if config.VERBOSE:
        print(f"\n{model.name} Evaluation:")
        print(f"  MAE: ${mae:.2f}")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
    
    return results


def forecast_future_prophet(
    model: ProphetModel,
    horizon: Optional[int] = None,
    last_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Generate future forecasts with confidence intervals.
    
    Parameters
    ----------
    model : ProphetModel
        Fitted Prophet model
    horizon : int, optional
        Number of periods to forecast (default from config)
    last_date : pd.Timestamp, optional
        Last date in training data for generating future dates
    """
    horizon = horizon or config.PREDICTION_HORIZON_DAYS
    
    # Get forecasts with confidence intervals
    forecast_df = model.predict_with_confidence(horizon)
    
    # Add dates
    if last_date is None:
        last_date = model.train_data.index[-1]
    
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
    forecast_df.index = future_dates
    
    if config.VERBOSE:
        print(f"\n{model.name} - {horizon}-day Forecast:")
        print(f"Confidence Level: {model.interval_width * 100:.0f}%")
        print(forecast_df.head(10).to_string())
        if horizon > 10:
            print(f"... ({horizon - 10} more rows)")
    
    return forecast_df


def tune_prophet_hyperparameters(
    data: pd.DataFrame,
    column: str = 'Close',
    param_grid: Optional[Dict[str, List]] = None,
    train_ratio: float = 0.8
) -> Dict[str, Any]:
    """
    Tune Prophet hyperparameters using grid search.
    
    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    column : str
        Column to forecast
    param_grid : dict, optional
        Parameter grid for tuning
    train_ratio : float
        Train/test split ratio
    """
    if param_grid is None:
        param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
            'seasonality_prior_scale': [0.1, 1.0, 10.0],
            'seasonality_mode': ['additive', 'multiplicative']
        }
    
    # Split data
    train, test = train_test_split(data, train_ratio)
    
    best_mae = float('inf')
    best_params = {}
    all_results = []
    
    if config.VERBOSE:
        print("\n" + "=" * 50)
        print("PROPHET HYPERPARAMETER TUNING")
        print("=" * 50)
        
        # Calculate total combinations
        total = 1
        for values in param_grid.values():
            total *= len(values)
        print(f"Total combinations: {total}")
    
    # Grid search
    from itertools import product
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for i, values in enumerate(product(*param_values)):
        params = dict(zip(param_names, values))
        
        try:
            # Create and fit model with current params
            model = ProphetModel(**params)
            
            # Suppress output during tuning
            original_verbose = config.VERBOSE
            config.VERBOSE = False
            
            model.fit(train, column)
            
            # Evaluate
            eval_result = evaluate_prophet(model, test, column)
            
            config.VERBOSE = original_verbose
            
            result = {
                **params,
                'mae': eval_result['mae'],
                'rmse': eval_result['rmse'],
                'mape': eval_result['mape']
            }
            all_results.append(result)
            
            if eval_result['mae'] < best_mae:
                best_mae = eval_result['mae']
                best_params = params.copy()
                
        except Exception as e:
            if config.VERBOSE:
                print(f"Failed with params {params}: {e}")
            continue
    
    if config.VERBOSE:
        print(f"\nBest Parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"\nBest MAE: ${best_mae:.2f}")
    
    return {
        'best_params': best_params,
        'best_mae': best_mae,
        'all_results': pd.DataFrame(all_results)
    }


def run_prophet_analysis(
    data: pd.DataFrame,
    column: str = 'Close',
    train_ratio: Optional[float] = None,
    tune_hyperparameters: bool = False
) -> Dict[str, Any]:
    """
    Run complete Prophet analysis pipeline.
    """
    print("\n" + "=" * 60)
    print("PROPHET MODEL ANALYSIS")
    print("=" * 60)
    
    # Split data
    train, test = train_test_split(data, train_ratio)
    
    # Optionally tune hyperparameters
    if tune_hyperparameters:
        tuning_results = tune_prophet_hyperparameters(data, column, train_ratio=train_ratio or config.TRAIN_TEST_SPLIT)
        best_params = tuning_results['best_params']
    else:
        best_params = {}
    
    # Create and fit model
    model = ProphetModel(**best_params)
    model.fit(train, column)
    
    # Evaluate on test set
    evaluation = evaluate_prophet(model, test, column)
    
    # Get components
    components = model.get_components()
    
    # Get changepoints
    changepoints = model.get_changepoints()
    
    # Generate future forecast
    print("\n--- Future Forecast ---")
    forecast = forecast_future_prophet(
        model,
        horizon=config.PREDICTION_HORIZON_DAYS,
        last_date=data.index[-1]
    )
    
    # Compile results
    results = {
        'model': model,
        'train': train,
        'test': test,
        'evaluation': evaluation,
        'components': components,
        'changepoints': changepoints,
        'forecast': forecast,
        'params': model.get_params()
    }
    
    if tune_hyperparameters:
        results['tuning_results'] = tuning_results
    
    # Print summary
    print("\n" + "=" * 60)
    print("PROPHET ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Model: {model.name}")
    print(f"Training samples: {len(train)}")
    print(f"Test samples: {len(test)}")
    print(f"\nSeasonality Configuration:")
    print(f"  Mode: {model.seasonality_mode}")
    print(f"  Yearly: {model.yearly_seasonality}")
    print(f"  Weekly: {model.weekly_seasonality}")
    print(f"\nPerformance Metrics:")
    print(f"  MAE: ${evaluation['mae']:.2f}")
    print(f"  RMSE: ${evaluation['rmse']:.2f}")
    print(f"  MAPE: {evaluation['mape']:.2f}%")
    print(f"\nChangepoints detected: {len(changepoints)}")
    
    return results


if __name__ == "__main__":
    from src.data_cleaning import cleaned_data_loader
    
    print("=" * 60)
    print("Prophet Model - Example Usage")
    print("=" * 60)
    
    # Load data
    try:
        data = cleaned_data_loader()
    except FileNotFoundError:
        from src.data_collection import download_stock_data
        from src.data_cleaning import cleaned_data
        
        print("\nNo cleaned data found. Downloading and cleaning...")
        raw_data = download_stock_data()
        data = cleaned_data(raw_data)
    
    # Run Prophet analysis
    results = run_prophet_analysis(data, tune_hyperparameters=False)
    
    # Plot components
    print("\n--- Plotting Components ---")
    fig = results['model'].plot_components()
