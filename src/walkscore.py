"""
Walk Score API data collection.

Strategy (per proposal):
- Snap each listing's lat/lon to a ~110m grid to deduplicate coordinates
- Fetch walk/transit/bike scores from the Walk Score API (5,000 req/day limit)
- Cache all results to data/walkscore_cache.parquet so the API
  is never hit twice for the same snapped coordinate
- Already-cached coordinates are skipped on re-runs

Usage:
    python src/walkscore.py [--listings data/airbnb_cleaned.parquet] \
                            [--out data/walkscore_cache.parquet] \
                            [--daily-limit 4900] [--dry-run]

    Run collect_data.py then clean_data.py before this script.
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import polars as pl
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).parent / ".env")


## Coordinate snapping ##

# 1 degree of latitude ~ 111,320 m  ->  0.001 deg ~  111 m  (good enough imo)
GRID_DEG = 0.001


def snap(lat: float, lon: float, grid: float = GRID_DEG) -> tuple[float, float]:
    """Round lat/lon to the nearest grid cell center"""
    return (round(lat / grid) * grid, round(lon / grid) * grid)


def snapped_coords(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add snapped_lat / snapped_lon columns and return the unique coordinate set

    Expects columns: latitude, longitude (float)
    """
    return (
        df.select(["latitude", "longitude"])
        .with_columns(
            [
                (pl.col("latitude") / GRID_DEG).round(0) * GRID_DEG,
                (pl.col("longitude") / GRID_DEG).round(0) * GRID_DEG,
            ]
        )
        .rename({"latitude": "snapped_lat", "longitude": "snapped_lon"})
        .unique()
    )


## Walk Score API ##

WALKSCORE_URL = "https://api.walkscore.com/score"

# Fields returned by the API that we store.
SCORE_FIELDS = ["walkscore", "transit", "bike"]


def fetch_scores(lat: float, lon: float, api_key: str, address: str = "") -> dict:
    """
    Call the Walk Score API for a single coordinate

    Returns a dict with keys: snapped_lat, snapped_lon, walkscore,
    transit_score, bike_score, status, and raw_json (for debugging)

    Raises http error
    """
    params = {
        "format": "json",
        "lat": lat,
        "lon": lon,
        "address": address or f"{lat},{lon}",
        "wsapikey": api_key,
        "transit": 1,
        "bike": 1,
    }
    resp = requests.get(WALKSCORE_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    walk = data.get("walkscore")
    transit = (data.get("transit") or {}).get("score")
    bike = (data.get("bike") or {}).get("score")

    return {
        "snapped_lat": lat,
        "snapped_lon": lon,
        "walkscore": walk,
        "transit_score": transit,
        "bike_score": bike,
        "status": data.get("status"),
    }


## Cache helpers ##

def load_cache(path: Path) -> pl.DataFrame:
    """Load existing cache, or return an empty df with the right schema"""
    if path.exists():
        return pl.read_parquet(path)
    return pl.DataFrame(
        schema={
            "snapped_lat": pl.Float64,
            "snapped_lon": pl.Float64,
            "walkscore": pl.Int64,
            "transit_score": pl.Float64,
            "bike_score": pl.Float64,
            "status": pl.Int64,
        }
    )


def save_cache(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


## Listings loader ##

CITIES = ["nyc", "la", "chicago"]


def load_listings(cleaned_path: Path) -> pl.DataFrame:
    """
    Read latitude/longitude from the cleaned parquet produced by clean_data.py
    """
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"{cleaned_path} not found. Run clean_data.py first."
        )
    return pl.read_parquet(cleaned_path, columns=["latitude", "longitude"])


## Main collection loop ##

def collect(
    listings_path: Path,
    out_path: Path,
    api_key: str,
    daily_limit: int = 4900,
    dry_run: bool = False,
    delay: float = 0.25,
) -> None:
    """
    Full pipeline: load listings -> snap coords -> skip cached -> fetch -> save

    Stops when daily_limit is reached so you can re-run the next day to
    continue building up the cache
    """
    print("Loading listings…")
    listings = load_listings(listings_path)
    print(f"  {len(listings):,} total listing rows across all cities")

    unique_coords = snapped_coords(listings)
    print(f"  {len(unique_coords):,} unique snapped coordinates")

    cache = load_cache(out_path)
    print(f"  {len(cache):,} coordinates already cached")

    if len(cache) > 0:
        # Anti-join: keep only coords not yet in the cache
        remaining = unique_coords.join(
            cache.select(["snapped_lat", "snapped_lon"]),
            on=["snapped_lat", "snapped_lon"],
            how="anti",
        )
    else:
        remaining = unique_coords

    print(f"  {len(remaining):,} coordinates to fetch")

    if dry_run:
        print("[dry-run] Would fetch the above. Exiting.")
        return

    rows = []
    fetched = 0
    errors = 0

    total = min(len(remaining), daily_limit)
    try:
        for row in tqdm(remaining.iter_rows(named=True), total=total, unit="coord"):
            if fetched >= daily_limit:
                print(f"Daily limit of {daily_limit} reached. Re-run tomorrow.")
                break

            lat = row["snapped_lat"]
            lon = row["snapped_lon"]

            try:
                result = fetch_scores(lat, lon, api_key)
                rows.append(result)
                fetched += 1

                if fetched % 100 == 0:
                    # Flush to disk and reload so next flush includes all rows
                    _flush(rows, cache, out_path)
                    cache = load_cache(out_path)
                    rows = []

                time.sleep(delay)

            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 429:
                    print(f"\n[RATE LIMIT] Daily quota hit after {fetched} requests. Re-run tomorrow.")
                    break
                errors += 1
                if errors > 5:
                    print(f"\n[ERROR] Too many failures, aborting after {fetched} fetched.")
                    break

            except Exception as exc:  # noqa: BLE001
                errors += 1
                if errors > 5:
                    print(f"\n[ERROR] Too many failures, aborting after {fetched} fetched.")
                    break

    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] Saving progress ({fetched} fetched)…")

    # Final flush
    if rows:
        _flush(rows, cache, out_path)

    final_cache = load_cache(out_path)
    print(
        f"\nDone. {fetched} new records fetched, {errors} errors. "
        f"Cache now has {len(final_cache):,} rows -> {out_path}"
    )


def _flush(new_rows: list[dict], existing: pl.DataFrame, path: Path) -> None:
    """Append new_rows to existing cache and write to disk"""
    new_df = pl.DataFrame(new_rows).cast(
        {
            "snapped_lat": pl.Float64,
            "snapped_lon": pl.Float64,
            "walkscore": pl.Int64,
            "transit_score": pl.Float64,
            "bike_score": pl.Float64,
            "status": pl.Int64,
        }
    )
    combined = pl.concat([existing, new_df], how="diagonal")
    save_cache(combined, path)


## CLI ##

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Walk Score data for listings.")
    parser.add_argument(
        "--listings",
        type=Path,
        default=Path("data/airbnb_cleaned.parquet"),
        help="Path to cleaned listings parquet (output of clean_data.py)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/walkscore_cache.parquet"),
        help="Output parquet cache path",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("WALKSCORE_API_KEY", ""),
        help="Walk Score API key (or set WALKSCORE_API_KEY env var)",
    )
    parser.add_argument(
        "--daily-limit",
        type=int,
        default=4900,
        help="Max requests per run (default 4900, safely under the 5000/day free tier)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Seconds between requests (default 0.25)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be fetched without making API calls",
    )
    args = parser.parse_args()

    if not args.api_key and not args.dry_run:
        parser.error(
            "No API key found. Set WALKSCORE_API_KEY in src/.env or pass --api-key."
        )

    collect(
        listings_path=args.listings,
        out_path=args.out,
        api_key=args.api_key,
        daily_limit=args.daily_limit,
        dry_run=args.dry_run,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
