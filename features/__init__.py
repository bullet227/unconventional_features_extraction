# features/__init__.py
# Unconventional feature extraction modules

# Tier 1: Foundation Features
from .session_features import add_session_stats
from .stop_hunt import stop_hunt_metrics
from .sentiment_contra import retail_sentiment_features

# Tier 2: Behavioral/Technical
from .market_psychology import add_market_psychology_features
from .fibonacci_time import add_fibonacci_time_features
from .lunar_cycles import add_lunar_features

# Tier 3: Advanced Mathematical
from .chaos_theory import add_chaos_features
from .social_physics import add_social_physics_features
from .game_theory import add_game_theory_features

# Tier 4: Quantum & Neural
from .quantum_finance import add_quantum_features, add_quantum_field_theory_features
from .neural_oscillations import add_neural_oscillation_features, add_brainwave_entropy_features

# Tier 5: Microstructure
from .hft_scalping import add_hft_scalping_features, create_hft_features_realtime, add_order_flow_features, add_execution_signals
from .orderflow_mm_detection import OrderFlowAnalyzer, add_orderflow_features, RealTimeOrderFlow
from .intracandle_dynamics import analyze_candle_formation, IntraCandleState, add_intracandle_features, IntraCandleTracker

# Tier 5: Esoteric
from .image_features import candle_image_to_vector
from .astro_finance import add_astro_features, add_solar_activity_features, add_eclipse_features

__all__ = [
    # Tier 1: Foundation
    'add_session_stats',
    'stop_hunt_metrics',
    'retail_sentiment_features',
    # Tier 2: Behavioral/Technical
    'add_market_psychology_features',
    'add_fibonacci_time_features',
    'add_lunar_features',
    # Tier 3: Advanced Mathematical
    'add_chaos_features',
    'add_social_physics_features',
    'add_game_theory_features',
    # Tier 4: Quantum & Neural
    'add_quantum_features',
    'add_quantum_field_theory_features',
    'add_neural_oscillation_features',
    'add_brainwave_entropy_features',
    # Tier 5: Microstructure
    'add_hft_scalping_features',
    'create_hft_features_realtime',
    'add_order_flow_features',
    'add_execution_signals',
    'OrderFlowAnalyzer',
    'add_orderflow_features',
    'RealTimeOrderFlow',
    'analyze_candle_formation',
    'IntraCandleState',
    'add_intracandle_features',
    'IntraCandleTracker',
    # Tier 5: Esoteric
    'candle_image_to_vector',
    'add_astro_features',
    'add_solar_activity_features',
    'add_eclipse_features',
]