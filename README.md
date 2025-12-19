# Unconventional Features for Financial ML

This project implements unconventional feature engineering and ML pipeline for forex/trading data, combining traditional technical analysis with unorthodox methods from physics, psychology, and chaos theory.

## Quick Start

### Using Docker (Recommended)

```bash
# Clone and enter the repository
cd unconventional_features_extraction

# Copy and configure environment
cp .env.example .env

# Run with sample data (quick test)
docker-compose up --build

# Run with your PostgreSQL database
docker-compose run ml-pipeline python ml_pipeline/run_pipeline.py \
    --asset EURUSD --timeframe H1
```

### Local Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run pipeline with sample data
python ml_pipeline/run_pipeline.py --sample
```

## Docker Services

### Available Services

| Service | Description | Command |
|---------|-------------|---------|
| `ml-pipeline` | Main ML training/backtest pipeline | `docker-compose up ml-pipeline` |
| `feature-etl` | Feature extraction from database | `docker-compose --profile etl up` |
| `dev` | Interactive development shell | `docker-compose --profile dev run dev` |
| `postgres` | PostgreSQL database | Auto-starts with other services |

### Docker Commands

```bash
# Build and run the main pipeline
docker-compose up --build

# Run feature extraction ETL
docker-compose --profile etl up feature-etl

# Interactive development
docker-compose --profile dev run dev

# Run with monitoring (Prometheus + Grafana)
docker-compose --profile monitoring up

# Custom pipeline run
docker-compose run ml-pipeline python ml_pipeline/run_pipeline.py \
    --asset EURUSD,GBPUSD \
    --timeframe H1,H4 \
    --model xgboost \
    --features 100

# View logs
docker-compose logs -f ml-pipeline

# Clean up
docker-compose down -v
```

### Connecting to External Database

```bash
# Edit .env to point to your forex_trading_data database
POSTGRES_HOST=192.168.50.210
POSTGRES_PORT=5021
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password

# Run pipeline
docker-compose run ml-pipeline python ml_pipeline/run_pipeline.py \
    --asset EURUSD --timeframe H1
```

### Volume Mounts

| Local Path | Container Path | Purpose |
|------------|----------------|---------|
| `./data` | `/app/data` | Input data files |
| `./models` | `/app/models` | Trained models |
| `./reports` | `/app/reports` | HTML/JSON reports |
| `./logs` | `/app/logs` | Application logs |

## ML Pipeline

### Components

```
ml_pipeline/
├── __init__.py           # Exports all components
├── data_loader.py        # Database and data management
├── feature_processor.py  # Feature selection and scaling
├── model_trainer.py      # XGBoost/LightGBM with walk-forward CV
├── backtester.py         # Event-driven backtesting
├── evaluator.py          # Performance analysis and reports
└── run_pipeline.py       # CLI entry point
```

### Usage Example

```python
from ml_pipeline import (
    DataLoader, FeatureProcessor, ModelTrainer,
    Backtester, BacktestConfig, Evaluator
)
from unconventional_features import UnconventionalFeatureExtractor

# Load data
loader = DataLoader()
df = loader.load_ohlcv('EURUSD', 'H1')

# Extract 323 unconventional features
extractor = UnconventionalFeatureExtractor()
enriched_df = extractor.enrich(df, asset='EURUSD', timeframe='H1')

# Process features
processor = FeatureProcessor()
X = processor.handle_missing_values(features)
selection = processor.select_features(X, y, top_n=100)

# Train with walk-forward validation
trainer = ModelTrainer(model_type='xgboost', n_splits=5)
result = trainer.train(X, y)

# Backtest
config = BacktestConfig(spread_pips=1.0, stop_loss_pips=30)
bt_result = Backtester(config).run(prices, signals, timestamps)

# Generate report
Evaluator().save_report(report, format='html')
```

### CLI Options

```bash
python ml_pipeline/run_pipeline.py \
    --asset EURUSD \           # Currency pair
    --timeframe H1 \           # Timeframe
    --model xgboost \          # xgboost or lightgbm
    --ensemble \               # Use model ensemble
    --features 100 \           # Max features to select
    --splits 5 \               # Walk-forward splits
    --output reports \         # Output directory
    --sample                   # Use sample data (no DB required)
```

## Features Overview (323 Features in 5 Tiers)

### Tier 1: Foundation
- **Session Features**: Trading sessions, kill zones, day-of-week effects
- **Stop Hunt Detection**: Wick analysis, liquidity sweeps
- **Sentiment Analysis**: Retail positioning, contrarian indicators

### Tier 2: Behavioral/Technical
- **Market Psychology**: Fear/greed index, capitulation, herding
- **Fibonacci Time**: Time-based Fibonacci relationships
- **Lunar Cycles**: Moon phases, volatility correlation

### Tier 3: Advanced Mathematical
- **Chaos Theory**: Fractal dimensions, Lyapunov exponents, entropy
- **Social Physics**: Market momentum, energy, phase transitions
- **Game Theory**: Nash equilibrium, coordination patterns

### Tier 4: Quantum & Neural
- **Quantum Finance**: Wave functions, tunneling, superposition
- **Neural Oscillations**: EEG-inspired market rhythms

### Tier 5: Microstructure & Esoteric
- **HFT Scalping**: Microstructure analysis, execution signals
- **Order Flow**: Market maker detection, spoofing, liquidity
- **Intracandle Dynamics**: Real-time formation analysis
- **Astro Finance**: Planetary positions, solar activity

## Configuration

### Environment Variables

```bash
# Database
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=forex_ml_2024
FOREX_DB=forex_trading_data
UNCONVENTIONAL_DB=unconventional_features

# Pipeline
BATCH_SIZE=250000
PARALLEL_WORKERS=4
ENABLE_IMAGE_FEATURES=false
ENABLE_PROMETHEUS=false

# Model
MODEL_TYPE=xgboost
N_SPLITS=5
MAX_FEATURES=100

# Assets
ASSET_LIST=EURUSD,GBPUSD,USDJPY
TIMEFRAME_LIST=H1,H4,D
```

## Output Schema

Features are stored in PostgreSQL:
```
unconventional.{asset}_{timeframe}_features
```

Reports are generated as HTML/JSON in the `reports/` directory.

## Requirements

### Core Dependencies
- Python 3.11+
- polars, pandas, numpy
- xgboost, lightgbm, scikit-learn
- sqlalchemy, psycopg (for database)

### Optional Dependencies
- pyephem (astro features)
- torch, opencv-python (GPU image features)
- prometheus-client (monitoring)

## Research Rationale

These unconventional features capture market behavior that traditional indicators miss:

- **Non-correlation**: Uncorrelated with standard technical indicators
- **Behavioral Edge**: Psychological patterns in price action
- **Complex Systems View**: Markets as adaptive, chaotic systems
- **Alpha Generation**: Unique features for trading edge

## License

Proprietary - All rights reserved.
