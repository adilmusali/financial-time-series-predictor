# ============================================================================
# Data Collection
# ============================================================================

# Default stock ticker symbol (e.g., 'AAPL' for Apple, 'MSFT' for Microsoft)
DEFAULT_TICKER = 'AAPL'

# Date range for historical data collection
# Format: 'YYYY-MM-DD'
START_DATE = '2020-01-01'
END_DATE = None  # None means use current date

# Data interval (daily recommended for stock prices)
# Options: '1d', '1wk', '1mo'
DATA_INTERVAL = '1d'

# Default data source
# Options: 'yfinance', 'csv'
# When using 'csv', you must provide filepath parameter
DEFAULT_DATA_SOURCE = 'yfinance'

# ============================================================================
# Data Processing
# ============================================================================

# Train/Test split ratio (e.g., 0.8 means 80% train, 20% test)
TRAIN_TEST_SPLIT = 0.8

# Minimum data points required for training
MIN_DATA_POINTS = 100

# ============================================================================
# Prediction
# ============================================================================

# Prediction horizon in trading days
# 1 week = 5 trading days, 4 weeks = 20 trading days
PREDICTION_HORIZON_DAYS = 20  # 4 weeks ahead

# ============================================================================
# Baseline Model
# ============================================================================

# Moving average window size (in days)
MOVING_AVERAGE_WINDOW = 30

# ============================================================================
# ARIMA Model
# ============================================================================

# Auto-ARIMA settings
USE_AUTO_ARIMA = True

# (p, d, q) - Autoregressive, Differencing, Moving Average
ARIMA_ORDER = (5, 1, 0)

# Seasonal ARIMA parameters (optional)
# (P, D, Q, s) - Seasonal components
ARIMA_SEASONAL_ORDER = None  # e.g., (1, 1, 1, 5) for weekly seasonality

# ============================================================================
# Prophet Model
# ============================================================================

# Changepoint prior scale (controls flexibility of trend changes)
PROPHET_CHANGEPOINT_PRIOR_SCALE = 0.05

# Seasonality mode: 'additive' or 'multiplicative'
PROPHET_SEASONALITY_MODE = 'additive'

# Include yearly seasonality
PROPHET_YEARLY_SEASONALITY = True

# Include weekly seasonality
PROPHET_WEEKLY_SEASONALITY = True

# Include daily seasonality (usually False for daily stock data)
PROPHET_DAILY_SEASONALITY = False

# Uncertainty interval width (0.8 = 80% confidence interval)
PROPHET_INTERVAL_WIDTH = 0.8

# ============================================================================
# LSTM Model
# ============================================================================

# Sequence length (lookback window in days)
LSTM_SEQUENCE_LENGTH = 60

# Number of LSTM units in each layer
LSTM_UNITS = [50, 50]  # List of units for each LSTM layer

# Dropout rate for regularization
LSTM_DROPOUT = 0.2

# Training epochs
LSTM_EPOCHS = 50

# Batch size for training
LSTM_BATCH_SIZE = 32

# Learning rate
LSTM_LEARNING_RATE = 0.001

# Random seed for reproducibility
LSTM_RANDOM_SEED = 42

# ============================================================================
# Evaluation
# ============================================================================

# Metrics to calculate
EVALUATION_METRICS = ['MAE', 'RMSE', 'MAPE']

# ============================================================================
# Visualization
# ============================================================================

# Figure size for plots (width, height in inches)
FIGURE_SIZE = (12, 6)

# DPI for saved plots
PLOT_DPI = 300

# Plot style
PLOT_STYLE = 'seaborn-v0_8'

# ============================================================================
# File Paths
# ============================================================================

# Data directories
DATA_RAW_DIR = 'data/raw'
DATA_PROCESSED_DIR = 'data/processed'

# Output directories
PLOTS_DIR = 'plots'
MODELS_DIR = 'models'  # For saving trained models

# ============================================================================
# General Settings
# ============================================================================

# Random seed for reproducibility (used across all models)
RANDOM_SEED = 42

# Verbose output
VERBOSE = True
