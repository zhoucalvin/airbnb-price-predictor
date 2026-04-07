import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
INPUT_PATH  = DATA_DIR / "airbnb_combined.parquet"
OUTPUT_PATH = DATA_DIR / "airbnb_cleaned.parquet"


# helpers

def report(label: str, df: pd.DataFrame, prev_len: int) -> int:
    dropped = prev_len - len(df)
    print(f"  [{label}] dropped {dropped:,} rows  →  {len(df):,} remaining")
    return len(df)

# load data

def load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df

# clean the price

def clean_price(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    if "price_usd" not in df.columns:
        df["price_usd"] = pd.to_numeric(
            df["price"].astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .str.strip(),
            errors="coerce",
        )

    # drop missing / zero
    df = df[df["price_usd"].notna() & (df["price_usd"] > 0)]
    n = report("price missing/zero", df, n)

    # outlier cap per city
    def iqr_bounds(series: pd.Series, k: float = 3.0):
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        return q1 - k * iqr, q3 + k * iqr

    masks = []
    for city, grp in df.groupby("city"):
        lo, hi = iqr_bounds(grp["price_usd"])
        hi = max(hi, 50)   # never cap below $50
        lo = max(lo, 5)    # never allow below $5
        mask = (grp["price_usd"] >= lo) & (grp["price_usd"] <= hi)
        masks.append(mask)
        print(f"    {city}: IQR bounds ${lo:.0f} – ${hi:.0f}")
    df = df[pd.concat(masks).sort_index()]
    n = report("price IQR outliers", df, n)

    # log-transform (target variable)
    df["log_price"] = np.log1p(df["price_usd"])

    return df


def clean_numerics(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    # accommodates
    df = df[df["accommodates"].notna() & (df["accommodates"] >= 1)]
    n = report("accommodates missing/<1", df, n)

    # bedrooms & beds — fill with per-room_type median
    for col in ["bedrooms", "beds"]:
        if col in df.columns:
            medians = df.groupby("room_type")[col].transform("median")
            df[col] = df[col].fillna(medians).fillna(df[col].median())

    # minimum_nights cap
    if "minimum_nights" in df.columns:
        df["minimum_nights"] = df["minimum_nights"].clip(upper=365)

    # review scores — fill NaN with 0 (listing hasn't been reviewed yet)
    review_cols = [c for c in df.columns if c.startswith("review_scores_")]
    df[review_cols] = df[review_cols].fillna(0)

    if "number_of_reviews" in df.columns:
        df["number_of_reviews"] = df["number_of_reviews"].fillna(0)
    if "reviews_per_month" in df.columns:
        df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
    if "availability_365" in df.columns:
        df["availability_365"] = df["availability_365"].fillna(0)

    print(f"  Numeric fill complete.  Shape: {df.shape}")
    return df


def clean_bathrooms(df: pd.DataFrame) -> pd.DataFrame:
    def parse_bath(val):
        if pd.isna(val):
            return np.nan, 0
        val = str(val).lower()
        shared = int("shared" in val)
        if "half" in val:
            return 0.5, shared
        nums = re.findall(r"\d+\.?\d*", val)
        return (float(nums[0]) if nums else np.nan), shared

    if "bathrooms_text" in df.columns:
        parsed = df["bathrooms_text"].apply(parse_bath)
        df["bathrooms"]       = parsed.apply(lambda x: x[0])
        df["bathroom_shared"] = parsed.apply(lambda x: x[1]).astype(int)
        median_bath = df["bathrooms"].median()
        df["bathrooms"] = df["bathrooms"].fillna(median_bath)
    else:
        df["bathrooms"]       = 1.0
        df["bathroom_shared"] = 0

    print(f"  bathrooms range: {df['bathrooms'].min()} – {df['bathrooms'].max()}")
    return df


def clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    # room_type
    valid_rooms = {"Entire home/apt", "Private room", "Shared room", "Hotel room"}
    df["room_type"] = df["room_type"].where(df["room_type"].isin(valid_rooms), other="Other")
    print(f"  room_type distribution:\n{df['room_type'].value_counts().to_string()}")

    # property_type
    if "property_type" in df.columns:
        top_props = df["property_type"].value_counts().head(15).index
        df["property_type"] = df["property_type"].where(
            df["property_type"].isin(top_props), other="Other"
        ).fillna("Other")

    # boolean flags → 0/1
    bool_cols = ["host_is_superhost", "instant_bookable", "host_identity_verified"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.lower()
                .map({"t": 1, "true": 1, "f": 0, "false": 0})
                .fillna(0)
                .astype(int)
            )

    # neighbourhood
    if "neighbourhood_cleansed" in df.columns:
        df["neighbourhood_cleansed"] = df["neighbourhood_cleansed"].fillna("Unknown")

    return df

def engineer_host_tenure(df: pd.DataFrame) -> pd.DataFrame:
    if "host_since" in df.columns:
        reference = pd.Timestamp("2024-12-31")
        df["host_since_dt"] = pd.to_datetime(df["host_since"], errors="coerce")
        df["years_as_host"] = (
            (reference - df["host_since_dt"]).dt.days / 365.25
        ).clip(lower=0)
        df["years_as_host"] = df["years_as_host"].fillna(df["years_as_host"].median())
        df.drop(columns=["host_since_dt"], inplace=True)
        print(f"  years_as_host  mean={df['years_as_host'].mean():.1f}  max={df['years_as_host'].max():.1f}")
    return df


def parse_amenities(df: pd.DataFrame, min_freq: float = 0.01) -> pd.DataFrame:
    if "amenities" not in df.columns:
        print("  No amenities column found — skipping.")
        df["amenity_count"] = 0
        return df

    def parse_one(raw):
        if pd.isna(raw):
            return []
        try:
            val = json.loads(raw)
            if isinstance(val, list):
                return [str(a).strip().lower() for a in val]
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            import ast
            val = ast.literal_eval(str(raw))
            if isinstance(val, list):
                return [str(a).strip().lower() for a in val]
        except Exception:
            pass
        return []

    df["_amenity_list"] = df["amenities"].apply(parse_one)
    df["amenity_count"] = df["_amenity_list"].apply(len)

    # frequency count
    from collections import Counter
    all_amenities: Counter = Counter()
    for lst in df["_amenity_list"]:
        all_amenities.update(lst)

    threshold = int(min_freq * len(df))
    common = [a for a, cnt in all_amenities.items() if cnt >= threshold]
    print(f"  Total unique amenities : {len(all_amenities):,}")
    print(f"  Kept (>= {min_freq*100:.0f}% frequency): {len(common):,}")

    # sanitize column names
    def col_name(amenity: str) -> str:
        clean = re.sub(r"[^a-z0-9]+", "_", amenity).strip("_")
        return f"amenity_{clean[:40]}"

    amenity_df = pd.DataFrame(
        {col_name(a): df["_amenity_list"].apply(lambda lst, a=a: int(a in lst))
         for a in common},
        index=df.index,
    )
    df = pd.concat([df, amenity_df], axis=1)
    df.drop(columns=["_amenity_list", "amenities"], inplace=True)
    print(f"  Added {len(common)} amenity indicator columns.")
    return df


def final_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== 8. Final cleanup ===")
    n = len(df)

    # deduplicate on listing id + city
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id", "city"], keep="first")
        n = report("duplicate id+city", df, n)

    # drop raw/redundant columns
    drop_cols = ["price", "host_since", "name", "description", "bathrooms_text"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # drop fully-null columns
    all_null = [c for c in df.columns if df[c].isna().all()]
    if all_null:
        print(f"  Dropping fully-null columns: {all_null}")
        df.drop(columns=all_null, inplace=True)

    df = df.reset_index(drop=True)
    print(f"  Final shape: {df.shape}")
    return df

def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"  Total listings  : {len(df):,}")
    print(f"  Total columns   : {len(df.columns)}")
    print(f"\n  City breakdown:")
    print(df["city"].value_counts().to_string())
    print(f"\n  Price (USD) stats:")
    print(df["price_usd"].describe().round(2).to_string())
    print(f"\n  Null counts (top 10):")
    nulls = df.isna().sum().sort_values(ascending=False).head(10)
    print(nulls.to_string())
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    df = load(INPUT_PATH)

    df = clean_price(df)
    df = clean_numerics(df)
    df = clean_bathrooms(df)
    df = clean_categoricals(df)
    df = engineer_host_tenure(df)
    df = parse_amenities(df, min_freq=0.01)   # keep amenities in >= 1% of listings
    df = final_cleanup(df)

    print_summary(df)

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n✓  Cleaned data saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()