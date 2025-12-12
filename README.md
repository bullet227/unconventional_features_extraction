# Unconventional Features for Financial ML

This project implements unconventional feature engineering for forex/trading data, combining traditional technical analysis with unorthodox methods from physics, psychology, and chaos theory.

## Features Overview

### 1. **Session Features** (`features/session_features.py`)
- Trading session identification (Asian, London, NY)
- Session overlaps and kill zones
- Day-of-week effects (Monday gaps, Friday volatility)
- Intraday range percentages

### 2. **Stop Hunt Detection** (`features/stop_hunt.py`)
- Wick analysis and z-scores
- Upper/lower stop hunt detection
- Liquidity sweep identification
- Volume spike correlation

### 3. **Sentiment Analysis** (`features/sentiment_contra.py`)
- Retail positioning data
- Contrarian indicators
- Crowd sentiment skew

### 4. **Image Features** (`features/image_features.py`)
- CNN-based pattern recognition
- 512-dimensional feature vectors from candlestick patterns
- Visual pattern embedding using ResNet18

### 5. **Lunar Cycles** (`features/lunar_cycles.py`) 🌙
- Moon phase calculations
- Full/new moon indicators
- Lunar volatility factors
- Historical correlation with market turns

### 6. **Fibonacci Time** (`features/fibonacci_time.py`) 🔢
- Time-based Fibonacci relationships
- Swing high/low timing
- Fibonacci spiral projections
- Time symmetry detection

### 7. **Social Physics** (`features/social_physics.py`) ⚛️
- Market momentum as velocity
- Trading volume as mass
- Market kinetic/potential energy
- Force and pressure indicators
- Phase transitions and critical points
- Crowd coherence metrics

### 8. **Chaos Theory** (`features/chaos_theory.py`) 🦋
- Fractal dimensions
- Phase space reconstruction
- Lyapunov-like exponents
- Strange attractor analysis
- Entropy and predictability measures
- Bifurcation indicators

### 9. **Market Psychology** (`features/market_psychology.py`) 🧠
- Fear and greed indices
- Capitulation/euphoria detection
- Herding behavior strength
- Anchoring bias patterns
- Loss aversion indicators
- Cognitive load and confusion metrics

### 10. **Game Theory** (`features/game_theory.py`) 🎮
- Nash equilibrium detection
- Prisoner's dilemma patterns
- Zero-sum game indicators
- Coordination and chicken games
- Minimax strategies

### 11. **Quantum Finance** (`features/quantum_finance.py`) ⚛️
- Wave-particle duality
- Heisenberg uncertainty
- Quantum superposition states
- Entanglement correlation
- Quantum tunneling (breakouts)

### 12. **Neural Oscillations** (`features/neural_oscillations.py`) 🧠
- Delta, theta, alpha, beta, gamma waves
- Cross-frequency coupling
- Neural synchronization index
- Brain-wave inspired market rhythms

### 13. **HFT Scalping** (`features/hft_scalping.py`) ⚡
- Microstructure analysis
- Tick-level momentum
- Order flow imbalance
- Execution signals

### 14. **Order Flow & MM Detection** (`features/orderflow_mm_detection.py`)
- Market maker activity detection
- Spoofing detection
- Liquidity analysis
- Book imbalance metrics

### 15. **Intracandle Dynamics** (`features/intracandle_dynamics.py`)
- Real-time candle formation
- Path efficiency analysis
- Micro-reversal detection

### 16. **Astro Finance** (`features/astro_finance.py`) ⭐
- Planetary positions and aspects
- Solar activity correlation
- Eclipse features

## Configuration

### Environment Variables
```bash
# Database connections
POSTGRES_HOST=your_host
POSTGRES_PORT=5432
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
FOREX_DB=forex_data
UNCONVENTIONAL_DB=ml_features

# Processing
PARALLEL_WORKERS=4
BATCH_SIZE=250000
ASSET_LIST=ALL  # or comma-separated list
TIMEFRAME_LIST=ALL  # or comma-separated list

# Features
ENABLE_IMAGE_FEATURES=false  # Set to true for CNN features
SENTIMENT_API=http://your-sentiment-api  # Optional
```

## Usage

### Docker
```bash
docker-compose up --build
```

### Standalone
```bash
pip install -r requirements.txt
python unconventional_features.py
```

## Feature Rationale

### Why Lunar Cycles?
- Studies show correlation between moon phases and market volatility
- Full moons associated with reversal points
- Behavioral impact on trader psychology

### Why Social Physics?
- Markets behave like physical systems with momentum and inertia
- Volume represents mass, price change represents velocity
- Energy conservation principles apply to market moves

### Why Chaos Theory?
- Markets are complex adaptive systems
- Small changes can have large effects (butterfly effect)
- Strange attractors exist around key price levels

### Why These Unorthodox Methods?
1. **Non-correlation**: These features are uncorrelated with traditional indicators
2. **Behavioral Edge**: Capture psychological patterns others miss
3. **Systemic View**: Model markets as complex systems, not just price series
4. **Alpha Generation**: Unique features can provide trading edge

## Performance Considerations

- Image features are computationally expensive (disabled by default)
- Use parallel workers for multi-asset processing
- Batch size affects memory usage
- Some features require historical context (lookback periods)

## Output Schema

Features are written to PostgreSQL with schema:
```
unconventional.{asset}_{timeframe}_features
```

Each row contains:
- Original OHLCV data
- All calculated features (~100+ columns)
- Timestamp and instrument identifiers

## Research Notes

These unconventional features are based on:
- Academic research in behavioral finance
- Physics applications to social systems
- Chaos theory in financial markets
- Historical market anomalies
- Crowd psychology studies

While some features may seem esoteric, they're grounded in observable market phenomena and designed to capture aspects of market behavior that traditional indicators miss.

## License

Proprietary - All rights reserved.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
