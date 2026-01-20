"""LSTM model implementation for time series forecasting."""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

from src.models.baseline import train_test_split


def set_random_seed(seed: int = None) -> None:
    """Set random seed for reproducibility."""
    seed = seed or config.LSTM_RANDOM_SEED
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_sequences(
    data: np.ndarray,
    sequence_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.
    
    Parameters
    ----------
    data : np.ndarray
        Scaled time series data (1D or 2D)
    sequence_length : int
        Number of time steps to look back
        
    Returns
    -------
    X : np.ndarray
        Input sequences of shape (samples, sequence_length, features)
    y : np.ndarray
        Target values of shape (samples,)
    """
    X, y = [], []
    
    # Ensure data is 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length, 0])  # Predict first feature (Close price)
    
    return np.array(X), np.array(y)


def prepare_lstm_data(
    train_data: pd.DataFrame,
    test_data: Optional[pd.DataFrame] = None,
    column: str = 'Close',
    sequence_length: int = None,
    scaler_type: str = 'minmax'
) -> Dict[str, Any]:
    """
    Prepare data for LSTM model with scaling and sequence creation.
    
    Parameters
    ----------
    train_data : pd.DataFrame
        Training data
    test_data : pd.DataFrame, optional
        Test data (if provided, scaled using train scaler)
    column : str
        Column to use for prediction
    sequence_length : int
        Lookback window size
    scaler_type : str
        Type of scaler: 'minmax' or 'standard'
        
    Returns
    -------
    Dict containing X_train, y_train, X_test, y_test, scaler, and metadata
    """
    sequence_length = sequence_length or config.LSTM_SEQUENCE_LENGTH
    
    # Extract values
    train_values = train_data[column].values.reshape(-1, 1)
    
    # Create scaler
    if scaler_type == 'minmax':
        scaler = MinMaxScaler(feature_range=(0, 1))
    else:
        scaler = StandardScaler()
    
    # Fit scaler on training data only
    train_scaled = scaler.fit_transform(train_values)
    
    # Create training sequences
    X_train, y_train = create_sequences(train_scaled, sequence_length)
    
    result = {
        'X_train': X_train,
        'y_train': y_train,
        'scaler': scaler,
        'sequence_length': sequence_length,
        'train_values': train_values,
        'train_scaled': train_scaled
    }
    
    # Process test data if provided
    if test_data is not None:
        # For test data, we need the last sequence_length values from training
        # to create the first test sequence
        combined_values = np.concatenate([train_values, test_data[column].values.reshape(-1, 1)])
        combined_scaled = scaler.transform(combined_values)
        
        # Create sequences from the point where test starts
        test_start_idx = len(train_values) - sequence_length
        test_data_scaled = combined_scaled[test_start_idx:]
        
        X_test, y_test = create_sequences(test_data_scaled, sequence_length)
        
        result['X_test'] = X_test
        result['y_test'] = y_test
        result['test_values'] = test_data[column].values
    
    return result


def build_lstm_model(
    sequence_length: int,
    n_features: int = 1,
    lstm_units: List[int] = None,
    dropout_rate: float = None,
    learning_rate: float = None
) -> Sequential:
    """
    Build LSTM neural network model.
    
    Parameters
    ----------
    sequence_length : int
        Input sequence length (lookback window)
    n_features : int
        Number of input features
    lstm_units : List[int]
        Number of units in each LSTM layer
    dropout_rate : float
        Dropout rate for regularization
    learning_rate : float
        Learning rate for optimizer
        
    Returns
    -------
    Sequential model ready for training
    """
    lstm_units = lstm_units or config.LSTM_UNITS
    dropout_rate = dropout_rate if dropout_rate is not None else config.LSTM_DROPOUT
    learning_rate = learning_rate or config.LSTM_LEARNING_RATE
    
    model = Sequential()
    
    # Input layer with first LSTM
    model.add(Input(shape=(sequence_length, n_features)))
    
    # Add LSTM layers
    for i, units in enumerate(lstm_units):
        # Return sequences for all but the last LSTM layer
        return_sequences = i < len(lstm_units) - 1
        
        model.add(LSTM(
            units=units,
            return_sequences=return_sequences,
            activation='tanh',
            recurrent_activation='sigmoid'
        ))
        
        # Add dropout after each LSTM layer
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model


class LSTMModel:
    """LSTM model for time series forecasting."""
    
    def __init__(
        self,
        sequence_length: int = None,
        lstm_units: List[int] = None,
        dropout_rate: float = None,
        learning_rate: float = None,
        epochs: int = None,
        batch_size: int = None,
        scaler_type: str = 'minmax',
        early_stopping: bool = True,
        patience: int = 10,
        random_seed: int = None
    ):
        """
        Initialize LSTM model.
        
        Parameters
        ----------
        sequence_length : int
            Lookback window size in days
        lstm_units : List[int]
            Number of units in each LSTM layer
        dropout_rate : float
            Dropout rate for regularization
        learning_rate : float
            Learning rate for Adam optimizer
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        scaler_type : str
            Type of scaler: 'minmax' or 'standard'
        early_stopping : bool
            Whether to use early stopping
        patience : int
            Patience for early stopping
        random_seed : int
            Random seed for reproducibility
        """
        self.sequence_length = sequence_length or config.LSTM_SEQUENCE_LENGTH
        self.lstm_units = lstm_units or config.LSTM_UNITS
        self.dropout_rate = dropout_rate if dropout_rate is not None else config.LSTM_DROPOUT
        self.learning_rate = learning_rate or config.LSTM_LEARNING_RATE
        self.epochs = epochs or config.LSTM_EPOCHS
        self.batch_size = batch_size or config.LSTM_BATCH_SIZE
        self.scaler_type = scaler_type
        self.early_stopping = early_stopping
        self.patience = patience
        self.random_seed = random_seed or config.LSTM_RANDOM_SEED
        
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.train_data = None
        self.column = None
        self.history = None
        self.last_sequence = None
        
        self._name = f"LSTM (units={self.lstm_units}, seq={self.sequence_length})"
    
    @property
    def name(self) -> str:
        """Return model name."""
        return self._name
    
    def fit(
        self,
        train_data: pd.DataFrame,
        column: str = 'Close',
        validation_split: float = 0.1,
        verbose: int = None
    ) -> 'LSTMModel':
        """
        Fit LSTM model to training data.
        
        Parameters
        ----------
        train_data : pd.DataFrame
            Training data with datetime index
        column : str
            Column to forecast
        validation_split : float
            Fraction of training data to use for validation
        verbose : int
            Verbosity level for training (0=silent, 1=progress bar, 2=one line per epoch)
            
        Returns
        -------
        self
        """
        # Set random seed for reproducibility
        set_random_seed(self.random_seed)
        
        self.train_data = train_data.copy()
        self.column = column
        
        if config.VERBOSE:
            print(f"\n--- Fitting LSTM Model ---")
            print(f"Training samples: {len(train_data)}")
            print(f"Date range: {train_data.index.min().strftime('%Y-%m-%d')} to {train_data.index.max().strftime('%Y-%m-%d')}")
            print(f"Sequence length: {self.sequence_length}")
            print(f"LSTM units: {self.lstm_units}")
            print(f"Dropout rate: {self.dropout_rate}")
            print(f"Epochs: {self.epochs}")
            print(f"Batch size: {self.batch_size}")
        
        # Prepare data
        data_prep = prepare_lstm_data(
            train_data=train_data,
            column=column,
            sequence_length=self.sequence_length,
            scaler_type=self.scaler_type
        )
        
        X_train = data_prep['X_train']
        y_train = data_prep['y_train']
        self.scaler = data_prep['scaler']
        
        if config.VERBOSE:
            print(f"\nTraining sequences: {X_train.shape[0]}")
            print(f"Input shape: {X_train.shape}")
        
        # Build model
        self.model = build_lstm_model(
            sequence_length=self.sequence_length,
            n_features=1,
            lstm_units=self.lstm_units,
            dropout_rate=self.dropout_rate,
            learning_rate=self.learning_rate
        )
        
        # Callbacks
        callbacks = []
        if self.early_stopping:
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                restore_best_weights=True,
                verbose=1 if config.VERBOSE else 0
            ))
            callbacks.append(ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.patience // 2,
                min_lr=1e-6,
                verbose=1 if config.VERBOSE else 0
            ))
        
        # Determine verbosity
        if verbose is None:
            verbose = 1 if config.VERBOSE else 0
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Store last sequence for forecasting
        train_scaled = data_prep['train_scaled']
        self.last_sequence = train_scaled[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        
        self.is_fitted = True
        
        if config.VERBOSE:
            final_loss = self.history.history['loss'][-1]
            final_val_loss = self.history.history.get('val_loss', [None])[-1]
            print(f"\nModel trained successfully!")
            print(f"Final training loss: {final_loss:.6f}")
            if final_val_loss:
                print(f"Final validation loss: {final_val_loss:.6f}")
            print(f"Total epochs trained: {len(self.history.history['loss'])}")
        
        return self
    
    def predict(self, horizon: int) -> pd.Series:
        """
        Generate out-of-sample forecasts.
        
        Parameters
        ----------
        horizon : int
            Number of periods to forecast
            
        Returns
        -------
        pd.Series of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = []
        current_sequence = self.last_sequence.copy()
        
        for _ in range(horizon):
            # Predict next value
            pred_scaled = self.model.predict(current_sequence, verbose=0)
            predictions.append(pred_scaled[0, 0])
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = pred_scaled[0, 0]
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        return pd.Series(predictions)
    
    def predict_with_confidence(
        self,
        horizon: int,
        n_simulations: int = 100,
        confidence_level: float = 0.8
    ) -> pd.DataFrame:
        """
        Generate forecasts with confidence intervals using Monte Carlo dropout.
        
        Note: This is an approximation. For true uncertainty, consider
        Bayesian approaches or ensemble methods.
        
        Parameters
        ----------
        horizon : int
            Number of periods to forecast
        n_simulations : int
            Number of Monte Carlo simulations
        confidence_level : float
            Confidence level for intervals (e.g., 0.8 for 80%)
            
        Returns
        -------
        DataFrame with forecast, lower, and upper columns
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get point forecast
        forecast = self.predict(horizon).values
        
        # Estimate uncertainty based on training residuals
        # This is a simple approach - more sophisticated methods exist
        train_values = self.train_data[self.column].values
        train_scaled = self.scaler.transform(train_values.reshape(-1, 1))
        
        X_train, y_train = create_sequences(train_scaled, self.sequence_length)
        predictions_train = self.model.predict(X_train, verbose=0).flatten()
        
        # Calculate residual standard deviation
        residuals = y_train - predictions_train
        residual_std_scaled = np.std(residuals)
        
        # Convert to original scale (approximate)
        scale_factor = self.scaler.data_range_[0] if hasattr(self.scaler, 'data_range_') else self.scaler.scale_[0]
        residual_std = residual_std_scaled * scale_factor
        
        # Calculate confidence intervals
        # Uncertainty grows with forecast horizon
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        # Expanding uncertainty with horizon
        horizon_multiplier = np.sqrt(np.arange(1, horizon + 1))
        
        lower = forecast - z_score * residual_std * horizon_multiplier
        upper = forecast + z_score * residual_std * horizon_multiplier
        
        result = pd.DataFrame({
            'forecast': forecast,
            'lower': lower,
            'upper': upper
        })
        
        return result
    
    def predict_in_sample(self, data: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Generate in-sample predictions (fitted values).
        
        Parameters
        ----------
        data : pd.DataFrame, optional
            Data to predict on (default: training data)
            
        Returns
        -------
        pd.Series of in-sample predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if data is None:
            data = self.train_data
        
        # Scale data
        values = data[self.column].values.reshape(-1, 1)
        scaled = self.scaler.transform(values)
        
        # Create sequences
        X, _ = create_sequences(scaled, self.sequence_length)
        
        # Predict
        predictions_scaled = self.model.predict(X, verbose=0)
        predictions = self.scaler.inverse_transform(predictions_scaled).flatten()
        
        # Align with original index (first sequence_length values have no prediction)
        full_predictions = np.full(len(data), np.nan)
        full_predictions[self.sequence_length:] = predictions
        
        return pd.Series(full_predictions, index=data.index)
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history."""
        if self.history is None:
            return {}
        return self.history.history
    
    def plot_training_history(self, figsize: Tuple[int, int] = (12, 4)):
        """
        Plot training and validation loss over epochs.
        
        Parameters
        ----------
        figsize : tuple
            Figure size
        """
        if self.history is None:
            raise ValueError("No training history available.")
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Loss plot
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE plot
        axes[1].plot(self.history.history['mae'], label='Training MAE')
        if 'val_mae' in self.history.history:
            axes[1].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[1].set_title('Model MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def summary(self) -> str:
        """Get model summary as string."""
        if self.model is None:
            return "Model not built yet."
        
        # Capture summary
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        return '\n'.join(stringlist)
    
    def save_model(self, filepath: str) -> None:
        """Save model to file."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.model.save(filepath)
        
        if config.VERBOSE:
            print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> 'LSTMModel':
        """Load model from file."""
        self.model = keras.models.load_model(filepath)
        self.is_fitted = True
        
        if config.VERBOSE:
            print(f"Model loaded from: {filepath}")
        
        return self
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'scaler_type': self.scaler_type,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'random_seed': self.random_seed
        }


def evaluate_lstm(
    model: LSTMModel,
    test_data: pd.DataFrame,
    column: str = 'Close'
) -> Dict[str, Any]:
    """
    Evaluate LSTM model on test data.
    
    Parameters
    ----------
    model : LSTMModel
        Fitted LSTM model
    test_data : pd.DataFrame
        Test data for evaluation
    column : str
        Column name to evaluate
        
    Returns
    -------
    Dict with evaluation metrics and predictions
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


def forecast_future_lstm(
    model: LSTMModel,
    horizon: Optional[int] = None,
    last_date: Optional[pd.Timestamp] = None,
    confidence_level: float = 0.8
) -> pd.DataFrame:
    """
    Generate future forecasts with confidence intervals.
    
    Parameters
    ----------
    model : LSTMModel
        Fitted LSTM model
    horizon : int, optional
        Number of periods to forecast (default from config)
    last_date : pd.Timestamp, optional
        Last date in data for generating future dates
    confidence_level : float
        Confidence level for intervals
        
    Returns
    -------
    DataFrame with forecast, lower, and upper columns indexed by date
    """
    horizon = horizon or config.PREDICTION_HORIZON_DAYS
    
    # Get forecasts with confidence intervals
    forecast_df = model.predict_with_confidence(horizon, confidence_level=confidence_level)
    
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


def tune_lstm_hyperparameters(
    data: pd.DataFrame,
    column: str = 'Close',
    param_grid: Optional[Dict[str, List]] = None,
    train_ratio: float = 0.8,
    verbose: int = 0
) -> Dict[str, Any]:
    """
    Tune LSTM hyperparameters using grid search.
    
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
    verbose : int
        Training verbosity
        
    Returns
    -------
    Dict with best parameters and all results
    """
    if param_grid is None:
        param_grid = {
            'sequence_length': [30, 60, 90],
            'lstm_units': [[32], [50, 50], [64, 32]],
            'dropout_rate': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.0001]
        }
    
    # Split data
    train, test = train_test_split(data, train_ratio)
    
    best_mae = float('inf')
    best_params = {}
    all_results = []
    
    if config.VERBOSE:
        print("\n" + "=" * 50)
        print("LSTM HYPERPARAMETER TUNING")
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
        
        if config.VERBOSE:
            print(f"\nCombination {i + 1}: {params}")
        
        try:
            # Create and fit model with current params
            model = LSTMModel(
                epochs=30,  # Reduced epochs for tuning
                early_stopping=True,
                patience=5,
                **params
            )
            
            # Suppress output during tuning
            original_verbose = config.VERBOSE
            config.VERBOSE = False
            
            model.fit(train, column, verbose=verbose)
            
            # Evaluate
            eval_result = evaluate_lstm(model, test, column)
            
            config.VERBOSE = original_verbose
            
            result = {
                **params,
                'mae': eval_result['mae'],
                'rmse': eval_result['rmse'],
                'mape': eval_result['mape']
            }
            all_results.append(result)
            
            if config.VERBOSE:
                print(f"  MAE: ${eval_result['mae']:.2f}, RMSE: ${eval_result['rmse']:.2f}")
            
            if eval_result['mae'] < best_mae:
                best_mae = eval_result['mae']
                best_params = params.copy()
                
        except Exception as e:
            if config.VERBOSE:
                print(f"  Failed: {e}")
            continue
    
    if config.VERBOSE:
        print(f"\n" + "=" * 50)
        print("TUNING COMPLETE")
        print("=" * 50)
        print(f"\nBest Parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"\nBest MAE: ${best_mae:.2f}")
    
    return {
        'best_params': best_params,
        'best_mae': best_mae,
        'all_results': pd.DataFrame(all_results)
    }


def run_lstm_analysis(
    data: pd.DataFrame,
    column: str = 'Close',
    train_ratio: Optional[float] = None,
    tune_hyperparameters: bool = False,
    **model_kwargs
) -> Dict[str, Any]:
    """
    Run complete LSTM analysis pipeline.
    
    Parameters
    ----------
    data : pd.DataFrame
        Time series data with datetime index
    column : str
        Column to forecast
    train_ratio : float, optional
        Train/test split ratio
    tune_hyperparameters : bool
        Whether to tune hyperparameters
    **model_kwargs
        Additional arguments passed to LSTMModel
        
    Returns
    -------
    Dict containing model, evaluation results, and forecast
    """
    print("\n" + "=" * 60)
    print("LSTM MODEL ANALYSIS")
    print("=" * 60)
    
    # Split data
    train, test = train_test_split(data, train_ratio)
    
    # Optionally tune hyperparameters
    if tune_hyperparameters:
        tuning_results = tune_lstm_hyperparameters(
            data, column,
            train_ratio=train_ratio or config.TRAIN_TEST_SPLIT
        )
        best_params = tuning_results['best_params']
        # Merge with any provided kwargs
        best_params.update(model_kwargs)
        model_kwargs = best_params
    
    # Create and fit model
    model = LSTMModel(**model_kwargs)
    model.fit(train, column)
    
    # Evaluate on test set
    evaluation = evaluate_lstm(model, test, column)
    
    # Generate future forecast
    print("\n--- Future Forecast ---")
    forecast = forecast_future_lstm(
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
        'forecast': forecast,
        'training_history': model.get_training_history(),
        'params': model.get_params()
    }
    
    if tune_hyperparameters:
        results['tuning_results'] = tuning_results
    
    # Print summary
    print("\n" + "=" * 60)
    print("LSTM ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Model: {model.name}")
    print(f"Training samples: {len(train)}")
    print(f"Test samples: {len(test)}")
    print(f"\nModel Configuration:")
    print(f"  Sequence length: {model.sequence_length}")
    print(f"  LSTM units: {model.lstm_units}")
    print(f"  Dropout rate: {model.dropout_rate}")
    print(f"  Learning rate: {model.learning_rate}")
    print(f"\nPerformance Metrics:")
    print(f"  MAE: ${evaluation['mae']:.2f}")
    print(f"  RMSE: ${evaluation['rmse']:.2f}")
    print(f"  MAPE: {evaluation['mape']:.2f}%")
    
    return results


if __name__ == "__main__":
    from src.data_cleaning import cleaned_data_loader
    
    print("=" * 60)
    print("LSTM Model - Example Usage")
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
    
    # Run LSTM analysis
    results = run_lstm_analysis(data, tune_hyperparameters=False)
    
    # Plot training history
    print("\n--- Plotting Training History ---")
    fig = results['model'].plot_training_history()
    
    # Print model summary
    print("\n--- Model Summary ---")
    print(results['model'].summary())
