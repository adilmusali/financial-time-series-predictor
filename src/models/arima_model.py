"""ARIMA model implementation for time series forecasting."""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union

# Suppress convergence warnings during model fitting
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

from src.models.baseline import train_test_split


def check_stationarity(
    series: pd.Series,
    significance_level: float = 0.05,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Check stationarity of a time series using ADF and KPSS tests.
    """
    results = {}
    
    # ADF Test (null hypothesis: series has unit root, i.e., non-stationary)
    adf_result = adfuller(series.dropna(), autolag='AIC')
    results['adf'] = {
        'statistic': adf_result[0],
        'p_value': adf_result[1],
        'lags_used': adf_result[2],
        'n_obs': adf_result[3],
        'critical_values': adf_result[4],
        'is_stationary': adf_result[1] < significance_level
    }
    
    # KPSS Test (null hypothesis: series is stationary)
    try:
        kpss_result = kpss(series.dropna(), regression='c', nlags='auto')
        results['kpss'] = {
            'statistic': kpss_result[0],
            'p_value': kpss_result[1],
            'lags_used': kpss_result[2],
            'critical_values': kpss_result[3],
            'is_stationary': kpss_result[1] > significance_level
        }
    except Exception as e:
        results['kpss'] = {'error': str(e), 'is_stationary': None}
    
    # Overall conclusion
    adf_stationary = results['adf']['is_stationary']
    kpss_stationary = results['kpss'].get('is_stationary', None)
    
    if adf_stationary and kpss_stationary:
        conclusion = "Stationary"
    elif not adf_stationary and not kpss_stationary:
        conclusion = "Non-stationary"
    elif adf_stationary and not kpss_stationary:
        conclusion = "Trend-stationary"
    else:
        conclusion = "Difference-stationary"
    
    results['conclusion'] = conclusion
    results['recommended_d'] = 0 if conclusion == "Stationary" else 1
    
    if verbose:
        print("\n--- Stationarity Tests ---")
        print(f"ADF Test:")
        print(f"  Statistic: {results['adf']['statistic']:.4f}")
        print(f"  p-value: {results['adf']['p_value']:.4f}")
        print(f"  Is Stationary: {results['adf']['is_stationary']}")
        if 'error' not in results['kpss']:
            print(f"KPSS Test:")
            print(f"  Statistic: {results['kpss']['statistic']:.4f}")
            print(f"  p-value: {results['kpss']['p_value']:.4f}")
            print(f"  Is Stationary: {results['kpss']['is_stationary']}")
        print(f"Conclusion: {conclusion}")
        print(f"Recommended d (differencing): {results['recommended_d']}")
    
    return results


def find_optimal_d(
    series: pd.Series,
    max_d: int = 2,
    significance_level: float = 0.05
) -> int:
    """
    Find optimal differencing order d for ARIMA.
    """
    for d in range(max_d + 1):
        if d == 0:
            diff_series = series
        else:
            diff_series = series.diff(d).dropna()
        
        adf_result = adfuller(diff_series, autolag='AIC')
        if adf_result[1] < significance_level:
            return d
    
    return max_d


def find_optimal_p_q(
    series: pd.Series,
    max_p: int = 5,
    max_q: int = 5,
    d: int = 1,
    criterion: str = 'aic'
) -> Tuple[int, int, float]:
    """
    Find optimal p and q parameters using grid search with information criterion.
    """
    best_score = float('inf')
    best_p, best_q = 0, 0
    
    if config.VERBOSE:
        print(f"\nSearching for optimal (p, q) with d={d}...")
        print(f"Testing p: 0-{max_p}, q: 0-{max_q}")
    
    results_grid = []
    
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue  # Skip (0, d, 0)
            
            try:
                model = ARIMA(series, order=(p, d, q))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted = model.fit()
                
                score = fitted.aic if criterion == 'aic' else fitted.bic
                results_grid.append({
                    'p': p, 'd': d, 'q': q,
                    'aic': fitted.aic, 'bic': fitted.bic
                })
                
                if score < best_score:
                    best_score = score
                    best_p, best_q = p, q
                    
            except Exception:
                continue
    
    if config.VERBOSE:
        print(f"Best parameters: p={best_p}, d={d}, q={best_q}")
        print(f"Best {criterion.upper()}: {best_score:.2f}")
    
    return best_p, best_q, best_score


def auto_arima(
    series: pd.Series,
    max_p: int = 5,
    max_d: int = 2,
    max_q: int = 5,
    criterion: str = 'aic',
    seasonal: bool = False,
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    stepwise: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Automatically find optimal ARIMA parameters.
    
        
    """
    if verbose:
        print("\n" + "=" * 50)
        print("AUTO-ARIMA PARAMETER SELECTION")
        print("=" * 50)
    
    # Step 1: Find optimal d
    optimal_d = find_optimal_d(series, max_d=max_d)
    if verbose:
        print(f"\nStep 1: Optimal differencing d = {optimal_d}")
    
    # Step 2: Find optimal p and q
    if stepwise:
        # Stepwise search (faster)
        best_p, best_q, best_score = _stepwise_search(
            series, optimal_d, max_p, max_q, criterion
        )
    else:
        # Full grid search
        best_p, best_q, best_score = find_optimal_p_q(
            series, max_p, max_q, optimal_d, criterion
        )
    
    if verbose:
        print(f"\nStep 2: Optimal (p, q) = ({best_p}, {best_q})")
        print(f"Best {criterion.upper()}: {best_score:.2f}")
    
    result = {
        'order': (best_p, optimal_d, best_q),
        'p': best_p,
        'd': optimal_d,
        'q': best_q,
        'criterion': criterion,
        'criterion_value': best_score,
        'seasonal': seasonal,
        'seasonal_order': seasonal_order
    }
    
    if verbose:
        print(f"\nOptimal ARIMA order: ({best_p}, {optimal_d}, {best_q})")
    
    return result


def _stepwise_search(
    series: pd.Series,
    d: int,
    max_p: int,
    max_q: int,
    criterion: str
) -> Tuple[int, int, float]:
    """
    Stepwise search for optimal p and q (faster than grid search).
    
    Starts from simple models and expands search space based on improvements.
    """
    # Start with simple models
    start_models = [(0, 1), (1, 0), (1, 1), (2, 1), (1, 2)]
    
    best_score = float('inf')
    best_p, best_q = 1, 0
    
    # Evaluate starting models
    for p, q in start_models:
        if p > max_p or q > max_q:
            continue
        try:
            model = ARIMA(series, order=(p, d, q))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = model.fit()
            score = fitted.aic if criterion == 'aic' else fitted.bic
            if score < best_score:
                best_score = score
                best_p, best_q = p, q
        except Exception:
            continue
    
    # Stepwise improvement
    improved = True
    visited = set()
    visited.add((best_p, best_q))
    
    while improved:
        improved = False
        # Try neighbors
        neighbors = [
            (best_p + 1, best_q), (best_p - 1, best_q),
            (best_p, best_q + 1), (best_p, best_q - 1),
            (best_p + 1, best_q + 1), (best_p - 1, best_q - 1)
        ]
        
        for p, q in neighbors:
            if (p, q) in visited or p < 0 or q < 0 or p > max_p or q > max_q:
                continue
            if p == 0 and q == 0:
                continue
                
            visited.add((p, q))
            
            try:
                model = ARIMA(series, order=(p, d, q))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted = model.fit()
                score = fitted.aic if criterion == 'aic' else fitted.bic
                
                if score < best_score:
                    best_score = score
                    best_p, best_q = p, q
                    improved = True
            except Exception:
                continue
    
    return best_p, best_q, best_score


class ARIMAModel:
    """ARIMA model for time series forecasting."""
    
    def __init__(
        self,
        order: Optional[Tuple[int, int, int]] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        use_auto: bool = True,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        criterion: str = 'aic'
    ):
        """
        Initialize ARIMA model.
        
        """
        self.order = order or config.ARIMA_ORDER
        self.seasonal_order = seasonal_order or config.ARIMA_SEASONAL_ORDER
        self.use_auto = use_auto if config.USE_AUTO_ARIMA else False
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.criterion = criterion
        
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        self.train_data = None
        self.column = None
        self.auto_params = None
        
        # Model name will be set after fitting
        self._base_name = "ARIMA"
    
    @property
    def name(self) -> str:
        """Return model name with parameters."""
        if self.is_fitted:
            return f"ARIMA{self.order}"
        return self._base_name
    
    def fit(
        self,
        train_data: pd.DataFrame,
        column: str = 'Close'
    ) -> 'ARIMAModel':
        """
        Fit ARIMA model to training data.

        """
        self.train_data = train_data.copy()
        self.column = column
        series = train_data[column]
        
        if config.VERBOSE:
            print(f"\n--- Fitting ARIMA Model ---")
            print(f"Training samples: {len(series)}")
            print(f"Date range: {series.index.min().strftime('%Y-%m-%d')} to {series.index.max().strftime('%Y-%m-%d')}")
        
        # Auto-parameter selection if enabled
        if self.use_auto:
            self.auto_params = auto_arima(
                series,
                max_p=self.max_p,
                max_d=self.max_d,
                max_q=self.max_q,
                criterion=self.criterion,
                verbose=config.VERBOSE
            )
            self.order = self.auto_params['order']
        
        # Fit the model
        if config.VERBOSE:
            print(f"\nFitting ARIMA{self.order}...")
        
        try:
            self.model = ARIMA(
                series,
                order=self.order,
                seasonal_order=self.seasonal_order
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.fitted_model = self.model.fit()
            
            self.is_fitted = True
            
            if config.VERBOSE:
                print(f"Model fitted successfully!")
                print(f"AIC: {self.fitted_model.aic:.2f}")
                print(f"BIC: {self.fitted_model.bic:.2f}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to fit ARIMA model: {str(e)}")
        
        return self
    
    def predict(self, horizon: int) -> pd.Series:
        """
        Generate out-of-sample forecasts.
        
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get forecast
        forecast = self.fitted_model.forecast(steps=horizon)
        
        return pd.Series(forecast.values)
    
    def predict_with_confidence(
        self,
        horizon: int,
        alpha: float = 0.05
    ) -> pd.DataFrame:
        """
        Generate forecasts with confidence intervals.
        
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get forecast with confidence intervals
        forecast_result = self.fitted_model.get_forecast(steps=horizon)
        forecast_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=alpha)
        
        # Create DataFrame
        result = pd.DataFrame({
            'forecast': forecast_mean.values,
            'lower': conf_int.iloc[:, 0].values,
            'upper': conf_int.iloc[:, 1].values
        })
        
        return result
    
    def predict_in_sample(self, data: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Generate in-sample predictions (fitted values).
        
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get fitted values
        fitted_values = self.fitted_model.fittedvalues
        
        if data is not None and len(data) != len(fitted_values):
            # Return predictions for the requested data length
            return pd.Series(fitted_values.values[-len(data):], index=data.index)
        
        return fitted_values
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get model diagnostics and statistics.
        
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        residuals = self.fitted_model.resid
        
        diagnostics = {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'log_likelihood': self.fitted_model.llf,
            'n_obs': self.fitted_model.nobs,
            'residual_mean': residuals.mean(),
            'residual_std': residuals.std(),
            'residual_skew': residuals.skew() if hasattr(residuals, 'skew') else None,
        }
        
        # AR and MA coefficients
        if self.order[0] > 0:  # AR terms
            diagnostics['ar_coefficients'] = self.fitted_model.arparams.tolist()
        if self.order[2] > 0:  # MA terms
            diagnostics['ma_coefficients'] = self.fitted_model.maparams.tolist()
        
        return diagnostics
    
    def summary(self) -> str:
        """Get model summary as string."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return str(self.fitted_model.summary())
    
    def plot_diagnostics(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot model diagnostics.
        
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        import matplotlib.pyplot as plt
        
        fig = self.fitted_model.plot_diagnostics(figsize=figsize)
        plt.suptitle(f'{self.name} Diagnostics', y=1.02)
        plt.tight_layout()
        
        return fig


def evaluate_arima(
    model: ARIMAModel,
    test_data: pd.DataFrame,
    column: str = 'Close'
) -> Dict[str, Any]:
    """
    Evaluate ARIMA model on test data.

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
        'order': model.order,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'predictions': predictions,
        'lower_bound': predictions_df['lower'].values,
        'upper_bound': predictions_df['upper'].values,
        'actuals': actuals,
        'errors': errors,
        'aic': model.fitted_model.aic,
        'bic': model.fitted_model.bic
    }
    
    if config.VERBOSE:
        print(f"\n{model.name} Evaluation:")
        print(f"  MAE: ${mae:.2f}")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  AIC: {model.fitted_model.aic:.2f}")
        print(f"  BIC: {model.fitted_model.bic:.2f}")
    
    return results


def forecast_future_arima(
    model: ARIMAModel,
    horizon: Optional[int] = None,
    last_date: Optional[pd.Timestamp] = None,
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Generate future forecasts with confidence intervals.
    
    """
    horizon = horizon or config.PREDICTION_HORIZON_DAYS
    alpha = 1 - confidence_level
    
    # Get forecasts with confidence intervals
    forecast_df = model.predict_with_confidence(horizon, alpha=alpha)
    
    # Add dates
    if last_date is None:
        future_dates = pd.RangeIndex(1, horizon + 1)
        forecast_df.index = future_dates
    else:
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
        forecast_df.index = future_dates
    
    if config.VERBOSE:
        print(f"\n{model.name} - {horizon}-day Forecast:")
        print(f"Confidence Level: {confidence_level * 100:.0f}%")
        print(forecast_df.head(10).to_string())
        if horizon > 10:
            print(f"... ({horizon - 10} more rows)")
    
    return forecast_df


def run_arima_analysis(
    data: pd.DataFrame,
    column: str = 'Close',
    train_ratio: Optional[float] = None,
    use_auto: bool = True,
    order: Optional[Tuple[int, int, int]] = None
) -> Dict[str, Any]:
    """
    Run complete ARIMA analysis pipeline.
    """
    print("\n" + "=" * 60)
    print("ARIMA MODEL ANALYSIS")
    print("=" * 60)
    
    # Split data
    train, test = train_test_split(data, train_ratio)
    
    # Check stationarity
    print("\n--- Stationarity Analysis ---")
    stationarity = check_stationarity(train[column], verbose=config.VERBOSE)
    
    # Create and fit model
    model = ARIMAModel(
        order=order,
        use_auto=use_auto
    )
    model.fit(train, column)
    
    # Evaluate on test set
    evaluation = evaluate_arima(model, test, column)
    
    # Generate future forecast
    print("\n--- Future Forecast ---")
    forecast = forecast_future_arima(
        model,
        horizon=config.PREDICTION_HORIZON_DAYS,
        last_date=data.index[-1]
    )
    
    # Compile results
    results = {
        'model': model,
        'train': train,
        'test': test,
        'stationarity': stationarity,
        'evaluation': evaluation,
        'forecast': forecast,
        'diagnostics': model.get_diagnostics()
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("ARIMA ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Model: {model.name}")
    print(f"Training samples: {len(train)}")
    print(f"Test samples: {len(test)}")
    print(f"\nPerformance Metrics:")
    print(f"  MAE: ${evaluation['mae']:.2f}")
    print(f"  RMSE: ${evaluation['rmse']:.2f}")
    print(f"  MAPE: {evaluation['mape']:.2f}%")
    print(f"\nModel Fit:")
    print(f"  AIC: {evaluation['aic']:.2f}")
    print(f"  BIC: {evaluation['bic']:.2f}")
    
    return results


if __name__ == "__main__":
    from src.data_cleaning import cleaned_data_loader
    
    print("=" * 60)
    print("ARIMA Model - Example Usage")
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
    
    # Run ARIMA analysis
    results = run_arima_analysis(data, use_auto=True)
    
    # Print model summary
    print("\n--- Model Summary ---")
    print(results['model'].summary())
