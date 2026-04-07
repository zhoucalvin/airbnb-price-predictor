import io
import warnings
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

INSIDE_AIRBNB_URLS = {
    "nyc": "https://data.insideairbnb.com/united-states/ny/new-york-city/2025-09-01/data/listings.csv.gz",
    "la":  "https://data.insideairbnb.com/united-states/ca/los-angeles/2025-09-01/data/listings.csv.gz",
    "chi": "https://data.insideairbnb.com/united-states/il/chicago/2025-09-22/data/listings.csv.gz",
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

    print(f"  [{city}] Downloading {url}…")
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
    df.to_csv(cache_path, index=False)
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


def save_combined(df: pd.DataFrame):
    out = DATA_DIR / "airbnb_combined.parquet"
    df.to_parquet(out, index=False)
    print(f"\n  ✓ Final dataset saved → {out}  ({len(df):,} rows × {len(df.columns)} columns)")
    print(f"    Columns: {list(df.columns)}")


def main():
    # args = parser.parse_args()

    # --- Inside Airbnb ---
    df = load_all_airbnb()
    # --- Save ---
    save_combined(df)

    print("Done!")


if __name__ == "__main__":
    main()