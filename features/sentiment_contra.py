# features/sentiment_contra.py
import polars as pl
import requests, os, json

BROKER_API = os.getenv("SENTIMENT_API", "")

def retail_sentiment_features(df: pl.DataFrame, asset: str) -> pl.DataFrame:
    try:
        resp = requests.get(f"{BROKER_API}/{asset}")
        skew  = json.loads(resp.text)["long_pct"]       # 0-100
    except Exception:
        skew = 50                                       
    contra = 100 - skew
    df = df.with_columns(pl.lit(skew).alias("retail_long_pct"),
                         pl.lit(contra).alias("contrarian_weight"))
    return df