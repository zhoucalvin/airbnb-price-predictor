import argparse
import io
import math
import os
import time
import warnings
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

INSIDE_AIRBNB_URLS = {
    "nyc": "http://data.insideairbnb.com/united-states/ny/new-york-city/2024-09-04/data/listings.csv.gz",
    "la":  "http://data.insideairbnb.com/united-states/ca/los-angeles/2024-09-12/data/listings.csv.gz",
    "chi": "http://data.insideairbnb.com/united-states/il/chicago/2024-09-15/data/listings.csv.gz",
}

KEEP_COLS = [
    "id", "name", "description",
    "neighbourhood_cleansed", "neighbourhood_group_cleansed",
    "latitude", "longitude",
    "room_type", "property_type",
    "accommodates", "bedrooms", "beds", "bathrooms_text",
    "amenities",
    "price",
    "minimum_nights", "maximum_nights",
    "number_of_reviews", "review_scores_rating",
    "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication",
    "review_scores_location", "review_scores_value",
    "host_id", "host_since", "host_is_superhost",
    "host_listings_count", "host_identity_verified",
    "instant_bookable",
    "availability_365",
    "calculated_host_listings_count",
    "reviews_per_month",
]

def download_city(city: str, url: str) -> pd.DataFrame:
    """Download a gzipped CSV from Inside Airbnb and return a DataFrame."""
    cache_path = DATA_DIR / f"{city}_raw.csv"
    if cache_path.exists():
        print(f"  [{city}] Cache hit → {cache_path}")
        return pd.read_csv(cache_path, low_memory=False)

    # check for a manually downloaded file
    for local_name in [f"{city}_listings.csv.gz", f"{city}_listings.csv"]:
        local_path = DATA_DIR / local_name
        if local_path.exists():
            print(f"  [{city}] Using local file → {local_path}")
            compression = "gzip" if local_name.endswith(".gz") else "infer"
            df = pd.read_csv(local_path, compression=compression, low_memory=False)
            # skip straight to column filtering below
            present = [c for c in KEEP_COLS if c in df.columns]
            missing = set(KEEP_COLS) - set(present)
            if missing:
                print(f"    Warning: columns not found in {city} snapshot: {missing}")
            df = df[present].copy()
            df["city"] = city
            df.to_csv(cache_path, index=False)
            print(f"    Saved {len(df):,} rows → {cache_path}")
            return df

    try:
        resp = requests.get(
            url,
            timeout=120,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/124.0.0.0 Safari/537.36",
                "Referer": "https://insideairbnb.com/",
            },
        )
        resp.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 403:
            raise SystemExit(
                f"\n[ERROR] 403 Forbidden for {city}.\n"
                "Inside Airbnb now blocks automated downloads.\n\n"
                "Manual fix (takes ~2 minutes):\n"
                "  1. Open https://insideairbnb.com/get-the-data/ in your browser\n"
                f"  2. Find the most recent '{city.upper()}' listing and download listings.csv.gz\n"
                f"  3. Rename it to  data/{city}_listings.csv.gz\n"
                "  4. Re-run this script.\n\n"
                "City → expected filename mapping:\n"
                "  nyc  →  data/nyc_listings.csv.gz\n"
                "  la   →  data/la_listings.csv.gz\n"
                "  chi  →  data/chi_listings.csv.gz\n"
            ) from exc
        raise

    df = pd.read_csv(
        io.BytesIO(resp.content),
        compression="gzip",
        low_memory=False,
    )

    # keep only the columns that exist in this snapshot
    present = [c for c in KEEP_COLS if c in df.columns]
    missing = set(KEEP_COLS) - set(present)
    if missing:
        print(f"    Warning: columns not found in {city} snapshot: {missing}")
    df = df[present].copy()

    df["city"] = city
    df.to_parquet(cache_path, index=False)
    print(f"    Saved {len(df):,} rows → {cache_path}")
    return df


def load_all_airbnb() -> pd.DataFrame:
    """Download all three cities and concatenate."""
    print("\n=== Step 1: Inside Airbnb ===")
    frames = []
    for city, url in INSIDE_AIRBNB_URLS.items():
        df = download_city(city, url)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Combined: {len(combined):,} listings across {combined['city'].nunique()} cities")
    return combined

def clean_price(series: pd.Series) -> pd.Series:
    """Strip '$' and ',' then cast to float."""
    return (
        series.astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.strip()
        .replace("", float("nan"))
        .astype(float)
    )

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df["price_usd"] = clean_price(df["price"])
    before = len(df)
    df = df[df["price_usd"].notna() & (df["price_usd"] > 0) & (df["price_usd"] <= 10_000)]
    print(f"  Dropped {before - len(df):,} rows (no price / price ≤ 0 / price > $10k)")
    print(f"  Remaining: {len(df):,} listings")
    df = df.reset_index(drop=True)
    return df

def save_combined(df: pd.DataFrame):
    out = DATA_DIR / "airbnb_combined.csv"
    df.to_csv(out, index=False)
    print(f"\n  ✓ Final dataset saved → {out}  ({len(df):,} rows × {len(df.columns)} columns)")
    print(f"    Columns: {list(df.columns)}")


def main():
    # args = parser.parse_args()

    # --- Inside Airbnb ---
    df = load_all_airbnb()

    # --- Basic clean ---
    df = basic_clean(df)

    # --- Save ---
    save_combined(df)

    print("Done!")


if __name__ == "__main__":
    main()