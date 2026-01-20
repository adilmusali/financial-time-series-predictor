# Financial Time Series Predictor

A comprehensive stock price forecasting system that implements multiple models (Naive, Moving Average, ARIMA, Prophet, LSTM) with automated data collection, preprocessing, evaluation, and visualization.

## Features

- **Data Collection**: Automatic download from Yahoo Finance or CSV import
- **Data Cleaning**: Missing value handling, outlier detection, validation
- **Exploratory Analysis**: Trend detection, seasonality, stationarity tests, ACF/PACF
- **Multiple Models**: Naive, Moving Average, ARIMA (auto), Prophet, LSTM
- **Evaluation**: MAE, RMSE, MAPE with model comparison
- **Visualization**: Predictions, forecasts, error distributions, dashboards
- **CLI Interface**: Easy command-line usage with configurable options

## Project Structure

```
time_forecast/
├── data/
│   ├── raw/                  # Downloaded raw data
│   └── processed/            # Cleaned data and forecasts
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── baseline_models.ipynb
│   ├── arima_model.ipynb
│   ├── prophet_model.ipynb
│   ├── lstm_model.ipynb
│   └── visualization.ipynb
├── plots/                    # Generated visualizations
├── src/
│   ├── data_collection.py
│   ├── data_cleaning.py
│   ├── exploratory_analysis.py
│   ├── evaluation.py
│   ├── visualization.py
│   ├── forecast.py           # Main pipeline
│   └── models/
│       ├── baseline.py
│       ├── arima_model.py
│       ├── prophet_model.py
│       └── lstm_model.py
├── config.py                 # Configuration parameters
├── main.py                   # CLI entry point
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd time_forecast
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify installation

```bash
python main.py --list-models
```

## Usage

### Quick Start

```bash
# Run with default settings (AAPL stock)
python main.py

# Forecast a specific stock
python main.py --ticker MSFT

# Skip LSTM for faster execution
python main.py --ticker GOOGL --no-lstm

# Include exploratory data analysis
python main.py --ticker TSLA --eda
```

### Command-Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--ticker` | `-t` | Stock ticker symbol | AAPL |
| `--start-date` | `-s` | Start date (YYYY-MM-DD) | 2020-01-01 |
| `--end-date` | `-e` | End date (YYYY-MM-DD) | today |
| `--horizon` | `-H` | Forecast days | 20 |
| `--train-ratio` | | Train/test split | 0.8 |
| `--no-lstm` | | Skip LSTM model | False |
| `--no-arima-auto` | | Manual ARIMA params | False |
| `--eda` | | Run EDA | False |
| `--csv` | | Load from CSV file | None |
| `--output-dir` | `-o` | Results directory | data/processed |
| `--plots-dir` | | Plots directory | plots |
| `--no-save` | | Don't save results | False |
| `--show-plots` | | Display plots | False |
| `--quiet` | `-q` | Suppress output | False |
| `--list-models` | | Show available models | - |

### Python API

```python
from src.forecast import ForecastingPipeline, run_forecast

# Quick run
results = run_forecast(ticker='AAPL', include_lstm=True)

# Step-by-step control
pipeline = ForecastingPipeline(
    ticker='MSFT',
    start_date='2022-01-01',
    prediction_horizon=30,
    verbose=True
)

pipeline.collect_data()
pipeline.clean_data()
pipeline.split_data()
pipeline.train_all_models(include_lstm=False)
pipeline.evaluate_models()
pipeline.generate_forecasts()
pipeline.visualize_results()
```

### Using Notebooks

Interactive notebooks are available in `notebooks/`:

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

## Models

### Baseline Models

- **Naive**: Uses the last observed value as prediction
- **Moving Average**: Simple moving average with configurable window (default: 30 days)

### Statistical Models

- **ARIMA**: AutoRegressive Integrated Moving Average
  - Auto-parameter selection using AIC criterion
  - Handles non-stationary data with differencing
  - Provides confidence intervals

- **Prophet**: Facebook's forecasting tool
  - Automatic trend and seasonality detection
  - Handles missing data and outliers
  - Configurable changepoint sensitivity

### Deep Learning

- **LSTM**: Long Short-Term Memory neural network
  - Sequence-based learning (60-day lookback)
  - Two LSTM layers with dropout regularization
  - Early stopping to prevent overfitting

## Evaluation Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **MAE** | Mean Absolute Error | Average prediction error in dollars |
| **RMSE** | Root Mean Squared Error | Penalizes large errors more heavily |
| **MAPE** | Mean Absolute Percentage Error | Error as percentage of actual value |

### Interpreting Results

- **Lower is better** for all metrics
- **MAE < $5**: Excellent for most stocks
- **MAPE < 5%**: Very good accuracy
- **MAPE 5-10%**: Acceptable for volatile stocks
- **MAPE > 10%**: Model may need tuning

## Output Files

After running the pipeline:

### Data Files (`data/processed/`)
- `cleaned_data_*.csv` - Preprocessed stock data
- `{TICKER}_all_forecasts_*.csv` - Combined forecasts from all models
- `{TICKER}_{model}_forecast_*.csv` - Individual model forecasts
- `{TICKER}_model_comparison_*.csv` - Metrics comparison table
- `evaluation_report_*.txt` - Text summary of results

### Plots (`plots/`)
- `model_comparison.png` - All models vs actual
- `metrics_comparison.png` - Bar chart of metrics
- `metrics_heatmap.png` - Performance heatmap
- `evaluation_dashboard.png` - Comprehensive dashboard
- `forecast_comparison.png` - Future predictions
- `{model}_predictions.png` - Individual predictions
- `{model}_errors.png` - Error distributions
- `{model}_residuals.png` - Residual analysis

## Configuration

Edit `config.py` to customize:

```python
# Data settings
DEFAULT_TICKER = 'AAPL'
START_DATE = '2020-01-01'
TRAIN_TEST_SPLIT = 0.8
PREDICTION_HORIZON_DAYS = 20

# ARIMA settings
USE_AUTO_ARIMA = True
ARIMA_ORDER = (5, 1, 0)

# Prophet settings
PROPHET_SEASONALITY_MODE = 'additive'

# LSTM settings
LSTM_SEQUENCE_LENGTH = 60
LSTM_UNITS = [50, 50]
LSTM_EPOCHS = 50
```

## Example Output

```
======================================================================
FORECAST SUMMARY
======================================================================

Best Model: ARIMA(2, 1, 1)
  MAE: $2.45
  RMSE: $3.12
  MAPE: 1.23%

20-Day Forecast (ARIMA):
  First day: $185.32
  Last day: $189.45

Results saved to:
  Data: data/processed/
  Plots: plots/
```

## Tips

1. **Speed**: Use `--no-lstm` for faster execution during testing
2. **Memory**: LSTM requires more RAM; reduce `LSTM_SEQUENCE_LENGTH` if needed
3. **Accuracy**: More historical data generally improves predictions
4. **Volatility**: High-volatility stocks will have higher MAPE values
5. **Overfitting**: If train metrics are much better than test, reduce model complexity

## Limitations

- Predictions are based on historical patterns only
- Does not account for news, earnings, or market events
- Short-term predictions (1-4 weeks) are more reliable
- Past performance does not guarantee future results

## License

MIT License
