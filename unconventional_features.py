#!/usr/bin/env python3
from __future__ import annotations
import os, logging, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import polars as pl
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from tqdm import tqdm
import prometheus_client as prom
import torch
import pyephem
from features.session_features import add_session_stats
from features.chaos_theory import add_chaos_features
from features.social_physics import add_social_physics_features
from features.lunar_cycles import add_lunar_features
from features.image_features import candle_image_to_vector
from features.market_psychology import add_market_psychology_features
from features.fibonacci_time import add_fibonacci_time_features
from features.stop_hunt import stop_hunt_metrics
from features.sentiment_contra import retail_sentiment_features
from features.orderflow_mm_detection import OrderFlowAnalyzer, add_orderflow_features
from features.intracandle_dynamics import analyze_candle_formation, IntraCandleState, add_intracandle_features
from features.hft_scalping import add_hft_scalping_features, create_hft_features_realtime
from features.quantum_finance import add_quantum_features, add_quantum_field_theory_features
from features.game_theory import add_game_theory_features
from features.neural_oscillations import add_neural_oscillation_features
from features.astro_finance import add_astro_features

# Logging (JSON)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("unconventional")
handler = logging.FileHandler("/var/log/unconventional/app.log")
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - %(extra)s'))
log.addHandler(handler)

# Prometheus
gauge_processed = prom.Gauge('rows_processed', 'Rows processed')
counter_errors = prom.Counter('errors_total', 'Total errors')
gauge_gpu_util = prom.Gauge('gpu_util', 'GPU utilization') if torch.cuda.is_available() else None
prom.start_http_server(8000)

# Env validation
def get_env(var, default=None):
    val = os.getenv(var, default)
    if val is None:
        raise ValueError(f"Missing env: {var}")
    return val

PG = dict(
    user=get_env("POSTGRES_USER", "postgres"),
    pwd=get_env("POSTGRES_PASSWORD", "Bullet227"),
    host=get_env("POSTGRES_HOST", "192.168.50.210"),
    port=get_env("POSTGRES_PORT", "5021"),
)
FOREX_DB = get_env("FOREX_DB", "forex_trading_data")
TARGET_DB = get_env("UNCONVENTIONAL_DB", "features_data")
BATCH_SIZE = int(get_env("BATCH_SIZE", 250000))
WORKERS = int(get_env("PARALLEL_WORKERS", 12))

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}", extra={"device": str(device)})

# Engines
def create_engine_with_retry(url, retries=5):
    for attempt in range(retries):
        try:
            return create_engine(url, pool_pre_ping=True, pool_size=20, max_overflow=40)
        except OperationalError as e:
            counter_errors.inc()
            log.error(f"DB connect fail (attempt {attempt+1}): {e}", extra={"error": str(e)})
            time.sleep(10)
    raise RuntimeError("DB connection failed")

read_url = f"postgresql+psycopg://{PG['user']}:{PG['pwd']}@{PG['host']}:{PG['port']}/{FOREX_DB}"
write_url = f"postgresql+psycopg://{PG['user']}:{PG['pwd']}@{PG['host']}:{PG['port']}/{TARGET_DB}"
read_eng = create_engine_with_retry(read_url)
write_eng = create_engine_with_retry(write_url)

def _stream(table: str):
    with read_eng.connect().execution_options(stream_results=True) as conn:
        try:
            for chunk in pd.read_sql(text(f"SELECT * FROM {table} ORDER BY time"), conn, chunksize=BATCH_SIZE):
                yield chunk
        except Exception as e:
            counter_errors.inc()
            log.error(f"Stream fail for {table}: {e}", extra={"table": table})

def _enrich(df: pd.DataFrame, asset: str, tf: str) -> pd.DataFrame:
    try:
        pl_df = pl.from_pandas(df)

        # Tier 1: Foundation Features (Session, Stop Hunt, Sentiment)
        pl_df = add_session_stats(pl_df)
        pl_df = stop_hunt_metrics(pl_df)
        pl_df = retail_sentiment_features(pl_df, asset)

        # Tier 2: Behavioral/Technical (Psychology, Fibonacci, Lunar)
        pl_df = add_market_psychology_features(pl_df)
        pl_df = add_fibonacci_time_features(pl_df)
        pl_df = add_lunar_features(pl_df)

        # Tier 3: Advanced Mathematical (Chaos, Social Physics, Game Theory)
        pl_df = add_chaos_features(pl_df)
        pl_df = add_social_physics_features(pl_df)
        pl_df = add_game_theory_features(pl_df)

        # Tier 4: Quantum & Neural (Quantum Finance, Neural Oscillations)
        pl_df = add_quantum_features(pl_df)
        pl_df = add_quantum_field_theory_features(pl_df)
        pl_df = add_neural_oscillation_features(pl_df)

        # Tier 5: Microstructure (HFT, Order Flow, Intracandle)
        pl_df = add_hft_scalping_features(pl_df)
        pl_df = add_orderflow_features(pl_df)
        pl_df = add_intracandle_features(pl_df)

        # Tier 5: Esoteric (Astro Finance, Image Features)
        pl_df = add_astro_features(pl_df)

        # GPU-accelerated image feature extraction
        if os.getenv("ENABLE_IMAGE_FEATURES", "false").lower() == "true":
            vecs = candle_image_to_vector(pl_df)
            pl_df = pl_df.with_columns(pl.Series("img_vec", vecs))

        gauge_processed.inc(len(df))
        return pl_df.to_pandas()
    except Exception as e:
        counter_errors.inc()
        log.error(f"Enrich fail: {e}", extra={"asset": asset, "tf": tf})
        return pd.DataFrame()

def _write(df: pd.DataFrame, asset: str, tf: str):
    if df.empty:
        return
    tbl = f"unconventional.{asset.lower()}_{tf}_features"
    try:
        df.to_sql(tbl, write_eng, schema="unconventional", if_exists="append", index=False, method="multi", chunksize=50000)
        log.info(f"Wrote {len(df)} rows to {tbl}", extra={"rows": len(df), "table": tbl})
    except Exception as e:
        counter_errors.inc()
        log.error(f"Write fail: {e}", extra={"table": tbl})

def _run(asset: str, tf: str):
    src = f"{asset.lower()}_{tf.lower()}_candles"
    log.info(f"Processing {asset} {tf}", extra={"asset": asset, "tf": tf})
    for chunk in tqdm(_stream(src), desc=f"{asset}_{tf}"):
        enriched = _enrich(chunk, asset, tf)
        _write(enriched, asset, tf)

def main():
    assets = get_env("ASSET_LIST", "ALL").split(",")
    tfs = get_env("TIMEFRAME_LIST", "ALL").split(",")

    if assets == ["ALL"]:
        with read_eng.connect() as conn:
            assets = [r[0] for r in conn.execute(text("SELECT DISTINCT instrument FROM instruments"))]

    if tfs == ["ALL"]:
        tfs = ["S5","S10","S15","S30","M1","M2","M4","M5","M10","M15","M30",
               "H1","H2","H3","H4","H6","H8","H12","D","W","M"]

    tasks = [(a, t) for a in assets for t in tfs]
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(_run, a, t) for a, t in tasks]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                counter_errors.inc()
                log.error(f"Task fail: {e}")

if __name__ == "__main__":
    main()