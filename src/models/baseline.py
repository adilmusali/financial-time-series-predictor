"""Baseline models for time series forecasting."""

import os
import sys
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


def train_test_split(
    data: pd.DataFrame,
    train_ratio: Optional[float] = None,
    split_date: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based train/test split (no shuffling)."""
    train_ratio = train_ratio or config.TRAIN_TEST_SPLIT
    
    if split_date:
        split_idx = data.index.get_loc(pd.Timestamp(split_date), method='nearest')
    else:
        split_idx = int(len(data) * train_ratio)
    
    train = data.iloc[:split_idx].copy()
    test = data.iloc[split_idx:].copy()
    
    if config.VERBOSE:
        print(f"Train/Test Split:")
        print(f"  Train: {len(train)} samples ({train.index.min().strftime('%Y-%m-%d')} to {train.index.max().strftime('%Y-%m-%d')})")
        print(f"  Test: {len(test)} samples ({test.index.min().strftime('%Y-%m-%d')} to {test.index.max().strftime('%Y-%m-%d')})")
    
    return train, test


class NaiveModel:
    """Naive forecasting model - uses last observed value as prediction."""
    
    def __init__(self):
        self.name = "Naive"
        self.last_value = None
        self.is_fitted = False
    
    def fit(self, train_data: pd.DataFrame, column: str = 'Close') -> 'NaiveModel':
        """Fit model by storing the last observed value."""
        self.last_value = train_data[column].iloc[-1]
        self.column = column
        self.is_fitted = True
        
        if config.VERBOSE:
            print(f"Naive Model fitted. Last value: ${self.last_value:.2f}")
        
        return self
    
    def predict(self, horizon: int) -> pd.Series:
        """Predict future values (constant = last observed)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = pd.Series([self.last_value] * horizon)
        return predictions
    
    def predict_in_sample(self, data: pd.DataFrame) -> pd.Series:
        """Generate in-sample predictions (shifted by 1)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Naive: prediction is previous day's value
        predictions = data[self.column].shift(1)
        return predictions


class SeasonalNaiveModel:
    """Seasonal naive model - uses value from same period in previous cycle."""
    
    def __init__(self, seasonal_period: int = 5):
        self.name = f"Seasonal Naive (period={seasonal_period})"
        self.seasonal_period = seasonal_period
        self.history = None
        self.is_fitted = False
    
    def fit(self, train_data: pd.DataFrame, column: str = 'Close') -> 'SeasonalNaiveModel':
        """Fit model by storing historical values."""
        self.history = train_data[column].copy()
        self.column = column
        self.is_fitted = True
        
        if config.VERBOSE:
            print(f"Seasonal Naive Model fitted. Period: {self.seasonal_period}")
        
        return self
    
    def predict(self, horizon: int) -> pd.Series:
        """Predict using values from seasonal_period ago."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = []
        for i in range(horizon):
            # Get value from seasonal_period ago (cycling through history)
            idx = -(self.seasonal_period - (i % self.seasonal_period))
            predictions.append(self.history.iloc[idx])
        
        return pd.Series(predictions)
    
    def predict_in_sample(self, data: pd.DataFrame) -> pd.Series:
        """Generate in-sample predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Use value from seasonal_period ago
        predictions = data[self.column].shift(self.seasonal_period)
        return predictions


class MovingAverageModel:
    """Simple Moving Average (SMA) forecasting model."""
    
    def __init__(self, window: Optional[int] = None):
        self.window = window or config.MOVING_AVERAGE_WINDOW
        self.name = f"Moving Average (window={self.window})"
        self.last_values = None
        self.is_fitted = False
    
    def fit(self, train_data: pd.DataFrame, column: str = 'Close') -> 'MovingAverageModel':
        """Fit model by storing last 'window' values."""
        self.last_values = train_data[column].iloc[-self.window:].values
        self.column = column
        self.history = train_data[column].copy()
        self.is_fitted = True
        
        current_ma = np.mean(self.last_values)
        if config.VERBOSE:
            print(f"Moving Average Model fitted. Window: {self.window}, Current MA: ${current_ma:.2f}")
        
        return self
    
    def predict(self, horizon: int) -> pd.Series:
        """Predict future values using rolling average."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = []
        values = list(self.last_values)
        
        for _ in range(horizon):
            # Predict as mean of last 'window' values
            pred = np.mean(values[-self.window:])
            predictions.append(pred)
            values.append(pred)
        
        return pd.Series(predictions)
    
    def predict_in_sample(self, data: pd.DataFrame) -> pd.Series:
        """Generate in-sample predictions using rolling mean."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Use rolling mean shifted by 1 (can't use current value to predict current)
        predictions = data[self.column].rolling(window=self.window).mean().shift(1)
        return predictions


class ExponentialMovingAverageModel:
    """Exponential Moving Average (EMA) forecasting model."""
    
    def __init__(self, span: Optional[int] = None, alpha: Optional[float] = None):
        if alpha is not None:
            self.alpha = alpha
            self.span = int(2 / alpha - 1)
        else:
            self.span = span or config.MOVING_AVERAGE_WINDOW
            self.alpha = 2 / (self.span + 1)
        
        self.name = f"EMA (span={self.span})"
        self.last_ema = None
        self.is_fitted = False
    
    def fit(self, train_data: pd.DataFrame, column: str = 'Close') -> 'ExponentialMovingAverageModel':
        """Fit model by computing EMA of training data."""
        self.column = column
        ema_series = train_data[column].ewm(span=self.span, adjust=False).mean()
        self.last_ema = ema_series.iloc[-1]
        self.history = train_data[column].copy()
        self.is_fitted = True
        
        if config.VERBOSE:
            print(f"EMA Model fitted. Span: {self.span}, Alpha: {self.alpha:.4f}, Last EMA: ${self.last_ema:.2f}")
        
        return self
    
    def predict(self, horizon: int) -> pd.Series:
        """Predict future values (EMA stays constant without new data)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # EMA forecast is constant (last EMA value)
        predictions = pd.Series([self.last_ema] * horizon)
        return predictions
    
    def predict_in_sample(self, data: pd.DataFrame) -> pd.Series:
        """Generate in-sample predictions using EMA."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Use EMA shifted by 1
        predictions = data[self.column].ewm(span=self.span, adjust=False).mean().shift(1)
        return predictions


class DriftModel:
    """Drift model - naive with trend adjustment."""
    
    def __init__(self):
        self.name = "Drift"
        self.last_value = None
        self.drift = None
        self.is_fitted = False
    
    def fit(self, train_data: pd.DataFrame, column: str = 'Close') -> 'DriftModel':
        """Fit model by computing average drift."""
        series = train_data[column]
        self.last_value = series.iloc[-1]
        self.first_value = series.iloc[0]
        n = len(series)
        
        # Drift = average change per period
        self.drift = (self.last_value - self.first_value) / (n - 1) if n > 1 else 0
        self.column = column
        self.is_fitted = True
        
        if config.VERBOSE:
            print(f"Drift Model fitted. Last: ${self.last_value:.2f}, Drift: ${self.drift:.4f}/day")
        
        return self
    
    def predict(self, horizon: int) -> pd.Series:
        """Predict with linear drift."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = [self.last_value + self.drift * (i + 1) for i in range(horizon)]
        return pd.Series(predictions)
    
    def predict_in_sample(self, data: pd.DataFrame) -> pd.Series:
        """Generate in-sample predictions with drift from first value."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        n = len(data)
        indices = np.arange(n)
        predictions = self.first_value + self.drift * indices
        return pd.Series(predictions, index=data.index)


def evaluate_baseline(
    model,
    test_data: pd.DataFrame,
    column: str = 'Close'
) -> Dict[str, Any]:
    """Evaluate baseline model on test data."""
    actuals = test_data[column].values
    predictions = model.predict(len(test_data)).values
    
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
        'actuals': actuals,
        'errors': errors
    }
    
    if config.VERBOSE:
        print(f"\n{model.name} Evaluation:")
        print(f"  MAE: ${mae:.2f}")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
    
    return results


def run_all_baselines(
    data: pd.DataFrame,
    column: str = 'Close',
    train_ratio: Optional[float] = None,
    ma_windows: List[int] = [7, 14, 30]
) -> Dict[str, Dict[str, Any]]:
    """Run all baseline models and compare results."""
    
    print("\n" + "=" * 60)
    print("BASELINE MODELS COMPARISON")
    print("=" * 60)
    
    # Split data
    train, test = train_test_split(data, train_ratio)
    
    results = {}
    
    # 1. Naive Model
    print("\n--- Naive Model ---")
    naive = NaiveModel()
    naive.fit(train, column)
    results['naive'] = evaluate_baseline(naive, test, column)
    results['naive']['model_obj'] = naive
    
    # 2. Seasonal Naive (weekly for trading days)
    print("\n--- Seasonal Naive Model ---")
    seasonal = SeasonalNaiveModel(seasonal_period=5)
    seasonal.fit(train, column)
    results['seasonal_naive'] = evaluate_baseline(seasonal, test, column)
    results['seasonal_naive']['model_obj'] = seasonal
    
    # 3. Drift Model
    print("\n--- Drift Model ---")
    drift = DriftModel()
    drift.fit(train, column)
    results['drift'] = evaluate_baseline(drift, test, column)
    results['drift']['model_obj'] = drift
    
    # 4. Moving Average Models
    for window in ma_windows:
        print(f"\n--- Moving Average (window={window}) ---")
        ma = MovingAverageModel(window=window)
        ma.fit(train, column)
        key = f'ma_{window}'
        results[key] = evaluate_baseline(ma, test, column)
        results[key]['model_obj'] = ma
    
    # 5. EMA Model
    print("\n--- Exponential Moving Average ---")
    ema = ExponentialMovingAverageModel(span=config.MOVING_AVERAGE_WINDOW)
    ema.fit(train, column)
    results['ema'] = evaluate_baseline(ema, test, column)
    results['ema']['model_obj'] = ema
    
    # Summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<30} {'MAE':>10} {'RMSE':>10} {'MAPE':>10}")
    print("-" * 60)
    
    for key, res in sorted(results.items(), key=lambda x: x[1]['mae']):
        print(f"{res['model']:<30} ${res['mae']:>9.2f} ${res['rmse']:>9.2f} {res['mape']:>9.2f}%")
    
    # Find best model
    best_model = min(results.items(), key=lambda x: x[1]['mae'])
    print(f"\nBest Model (by MAE): {best_model[1]['model']}")
    
    # Store train/test for reference
    results['_train'] = train
    results['_test'] = test
    
    return results


def forecast_future(
    model,
    horizon: Optional[int] = None,
    last_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """Generate future forecasts with dates."""
    horizon = horizon or config.PREDICTION_HORIZON_DAYS
    
    predictions = model.predict(horizon)
    
    if last_date is None:
        # Use generic index
        future_dates = pd.RangeIndex(1, horizon + 1)
    else:
        # Generate business days
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
    
    forecast_df = pd.DataFrame({
        'forecast': predictions.values
    }, index=future_dates)
    
    if config.VERBOSE:
        print(f"\n{model.name} - {horizon}-day Forecast:")
        print(forecast_df.head(10))
        if horizon > 10:
            print(f"... ({horizon - 10} more rows)")
    
    return forecast_df


if __name__ == "__main__":
    from src.data_cleaning import cleaned_data_loader
    
    print("=" * 60)
    print("Baseline Models - Example Usage")
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
    
    # Run all baselines
    results = run_all_baselines(data, ma_windows=[7, 14, 30])
    
    # Forecast with best model
    print("\n--- Future Forecast ---")
    best_key = min(
        [k for k in results.keys() if not k.startswith('_')],
        key=lambda k: results[k]['mae']
    )
    best_model = results[best_key]['model_obj']
    forecast = forecast_future(best_model, horizon=20, last_date=data.index[-1])
