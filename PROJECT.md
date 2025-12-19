# PROJECT.md - Unconventional Features ML Pipeline

## Project Overview

A complete ML pipeline for forex trading using 323 unconventional features extracted from OHLCV data. The system combines traditional technical analysis with unorthodox methods from physics, psychology, chaos theory, and quantum mechanics to generate alpha.

**Status:** ✅ Complete and Ready for Deployment

---

## 1. ML Pipeline Infrastructure

### Components Built

| Module | File | Purpose | Status |
|--------|------|---------|--------|
| **Data Loader** | `ml_pipeline/data_loader.py` | Load OHLCV from PostgreSQL, store/retrieve features | ✅ Complete |
| **Feature Processor** | `ml_pipeline/feature_processor.py` | Feature selection (correlation, XGBoost, SHAP), scaling | ✅ Complete |
| **Model Trainer** | `ml_pipeline/model_trainer.py` | XGBoost/LightGBM with walk-forward validation | ✅ Complete |
| **Backtester** | `ml_pipeline/backtester.py` | Event-driven backtest with realistic costs | ✅ Complete |
| **Evaluator** | `ml_pipeline/evaluator.py` | Performance analysis, HTML/JSON reports | ✅ Complete |

### Key Features

- **Walk-Forward Validation**: Expanding window cross-validation to prevent look-ahead bias
- **Ensemble Methods**: Combine XGBoost + LightGBM predictions
- **Feature Selection**: Correlation filtering, importance ranking, domain grouping
- **Realistic Backtesting**: Spread, slippage, commission, stop-loss/take-profit
- **Comprehensive Metrics**: Sharpe, Sortino, Calmar, profit factor, win rate, expectancy

### Usage

```python
from ml_pipeline import (
    DataLoader, FeatureProcessor, ModelTrainer,
    Backtester, BacktestConfig, Evaluator
)
from unconventional_features import UnconventionalFeatureExtractor

# Extract features
extractor = UnconventionalFeatureExtractor()
enriched_df = extractor.enrich(ohlcv_df, asset='EURUSD', timeframe='H1')

# Train model
trainer = ModelTrainer(model_type='xgboost', n_splits=5)
result = trainer.train(X, y)

# Backtest
config = BacktestConfig(spread_pips=1.0, stop_loss_pips=30)
bt_result = Backtester(config).run(prices, signals, timestamps)
```

---

## 2. Docker Containerization

### Container Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    docker-compose.yml                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  postgres   │  │ ml-pipeline │  │    feature-etl      │  │
│  │  (15-alpine)│  │  (python)   │  │  (profile: etl)     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │     dev     │  │ prometheus  │  │      grafana        │  │
│  │(profile:dev)│  │(monitoring) │  │   (monitoring)      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Files Created

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage build, Python 3.11, non-root user |
| `Dockerfile.gpu` | NVIDIA CUDA 12.1 support for GPU nodes |
| `docker-compose.yml` | Service orchestration with profiles |
| `docker-compose.gpu.yml` | GPU-enabled override configuration |
| `docker/init-databases.sh` | Creates both PostgreSQL databases |
| `docker/prometheus.yml` | Metrics scrape configuration |
| `.dockerignore` | Optimizes build context |
| `.env.example` | Environment variable template |
| `requirements.txt` | Python dependencies |

### Quick Start

```bash
# CPU-only deployment
docker-compose up --build

# GPU deployment (for nodes E-12, E-13, E-14)
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

---

## 3. Feature Modules (323 Features)

### 16 Modules Across 5 Tiers

| Tier | Module | Features | Status |
|------|--------|----------|--------|
| **1 - Foundation** | session_features.py | Session stats, kill zones | ✅ |
| | stop_hunt.py | Wick analysis, liquidity sweeps | ✅ |
| | sentiment_contra.py | Retail positioning, contrarian signals | ✅ |
| **2 - Behavioral** | market_psychology.py | Fear/greed, capitulation, herding | ✅ |
| | fibonacci_time.py | Time-based Fibonacci relationships | ✅ |
| | lunar_cycles.py | Moon phases, volatility correlation | ✅ |
| **3 - Mathematical** | chaos_theory.py | Fractal dimensions, Lyapunov, entropy | ✅ |
| | social_physics.py | Momentum, energy, phase transitions | ✅ |
| | game_theory.py | Nash equilibrium, coordination patterns | ✅ |
| **4 - Quantum/Neural** | quantum_finance.py | Wave functions, tunneling, superposition | ✅ |
| | neural_oscillations.py | EEG-inspired market rhythms | ✅ |
| **5 - Microstructure** | hft_scalping.py | Microstructure analysis | ✅ |
| | orderflow_mm_detection.py | Market maker detection, spoofing | ✅ |
| | intracandle_dynamics.py | Real-time candle formation | ✅ |
| **5 - Esoteric** | astro_finance.py | Planetary positions, solar activity | ✅ |
| | image_features.py | CNN candlestick pattern recognition | ✅ |

### Validation Results

```
Total Features Extracted: 323
All 16 modules integrated and tested
Polars API compatibility issues resolved
Optional dependencies handled gracefully (pyephem, torch, cv2)
```

---

## 4. GPU Node Deployment (E-12, E-13, E-14)

### Prerequisites

```bash
# Verify NVIDIA drivers and Docker runtime
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Deployment Steps

#### Step 1: Clone Repository
```bash
ssh user@e-12.local  # or e-13, e-14
cd /opt/ml-pipelines
git clone <repo-url> unconventional_features_extraction
cd unconventional_features_extraction
git checkout claude/unconventional-forex-features-7ZQ0y
```

#### Step 2: Configure Environment
```bash
cp .env.example .env
nano .env

# Set for GPU node:
POSTGRES_HOST=192.168.50.210
POSTGRES_PORT=5021
POSTGRES_USER=postgres
POSTGRES_PASSWORD=<your-password>
ENABLE_IMAGE_FEATURES=true    # Enable CNN features on GPU
PARALLEL_WORKERS=8            # Adjust based on GPU memory
```

#### Step 3: Build GPU Image
```bash
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml build
```

#### Step 4: Run Pipeline
```bash
# Interactive test
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml run ml-pipeline \
    python ml_pipeline/run_pipeline.py --asset EURUSD --timeframe H1

# Full ETL extraction
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml --profile etl up

# Background daemon
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

### Multi-Node Distribution

For distributed processing across E-12, E-13, E-14:

```bash
# E-12: Major pairs
ASSET_LIST=EURUSD,GBPUSD,USDJPY,USDCHF docker-compose up

# E-13: Minor pairs
ASSET_LIST=EURJPY,GBPJPY,AUDUSD,NZDUSD docker-compose up

# E-14: Exotic pairs
ASSET_LIST=USDMXN,USDZAR,USDTRY docker-compose up
```

### GPU Memory Management

| Timeframe | Batch Size | GPU Memory |
|-----------|------------|------------|
| M1, M5 | 50,000 | ~4GB |
| M15, M30 | 100,000 | ~6GB |
| H1, H4 | 250,000 | ~8GB |
| D, W | 500,000 | ~12GB |

---

## Project Statistics

| Metric | Value |
|--------|-------|
| Total Features | 323 |
| Feature Modules | 16 |
| ML Pipeline Modules | 5 |
| Lines of Code (Features) | ~3,500 |
| Lines of Code (ML Pipeline) | ~4,000 |
| Docker Services | 6 |
| Supported Models | XGBoost, LightGBM, Ensemble |

---

## Repository Structure

```
unconventional_features_extraction/
├── features/                    # 16 feature extraction modules
│   ├── session_features.py
│   ├── stop_hunt.py
│   ├── sentiment_contra.py
│   ├── market_psychology.py
│   ├── fibonacci_time.py
│   ├── lunar_cycles.py
│   ├── chaos_theory.py
│   ├── social_physics.py
│   ├── game_theory.py
│   ├── quantum_finance.py
│   ├── neural_oscillations.py
│   ├── hft_scalping.py
│   ├── orderflow_mm_detection.py
│   ├── intracandle_dynamics.py
│   ├── astro_finance.py
│   ├── image_features.py
│   └── __init__.py
├── ml_pipeline/                 # ML infrastructure
│   ├── data_loader.py
│   ├── feature_processor.py
│   ├── model_trainer.py
│   ├── backtester.py
│   ├── evaluator.py
│   ├── run_pipeline.py
│   └── __init__.py
├── docker/                      # Docker configuration
│   ├── init-databases.sh
│   └── prometheus.yml
├── unconventional_features.py   # Main orchestration
├── Dockerfile                   # CPU container
├── Dockerfile.gpu               # GPU container (CUDA 12.1)
├── docker-compose.yml           # Service orchestration
├── docker-compose.gpu.yml       # GPU override
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
├── .dockerignore
├── README.md
└── PROJECT.md                   # This file
```

---

## Git History

```
ac9cecc Add Docker containerization for ML pipeline deployment
e2d784c Add complete ML pipeline infrastructure for forex trading
4957afc Integrate all 16 feature modules and fix Polars API compatibility
e5fd666 Add astro finance features and tier configuration
ac153fc Add HFT, order flow, and intracandle dynamics features
8f951df Add quantum finance, game theory, and neural oscillations features
21b0b9a Add advanced features: chaos_theory, social_physics, market_psychology
```

---

## Next Steps

1. **Production Deployment**: Deploy to GPU nodes E-12, E-13, E-14
2. **Feature Validation**: Run against live forex_trading_data
3. **Model Optimization**: Hyperparameter tuning, feature selection refinement
4. **CNN Integration**: Enable image features with GPU acceleration
5. **Strategy Development**: Design trading strategies based on feature importance

---

**Branch:** `claude/unconventional-forex-features-7ZQ0y`
**Last Updated:** 2024-12-19
