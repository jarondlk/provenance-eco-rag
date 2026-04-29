"""
Remote sensing (satellite SST) preprocessing pipeline.

Handles two data sources:
  1. Pre-processed NetCDF subsets (onagawa_sst_subset/) – primary path
  2. Raw Himawari .DAT files (himawari_test_unzipped/) – optional, via satpy

Pipeline:
    list_sst_files → extract_point_timeseries / compute_daily_summary
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def list_sst_files(root: Path, pattern: str = "*.nc") -> List[Path]:
    """Discover NetCDF SST files under *root* (recursive, sorted)."""
    files = sorted(root.rglob(pattern))
    logger.info("Found %d SST files under %s", len(files), root)
    return files


def parse_sst_time_from_filename(path: Path) -> pd.Timestamp:
    """
    Extract UTC timestamp from an SST NetCDF filename.

    Expected pattern: ``onagawa_sst_YYYYMMDD_HHMM.nc``
    """
    m = re.search(r"(\d{8})_(\d{4})", path.name)
    if not m:
        raise ValueError(f"Could not parse timestamp from filename: {path.name}")
    return pd.to_datetime(f"{m.group(1)} {m.group(2)}", format="%Y%m%d %H%M")


# ---------------------------------------------------------------------------
# Point time series  (extracted from app_v0_1_ocean.py lines 268-303)
# ---------------------------------------------------------------------------
def extract_point_timeseries(
    root: Path,
    target_lat: float,
    target_lon: float,
) -> pd.DataFrame:
    """
    For each NetCDF file under *root*, extract SST at the grid point
    nearest to (target_lat, target_lon).

    Returns DataFrame with columns:
        file, time_utc, time_jst, sst, nearest_lat, nearest_lon
    """
    import xarray as xr

    files = list_sst_files(root)
    rows: List[Dict[str, Any]] = []

    for fp in files:
        ds = xr.open_dataset(str(fp), engine="netcdf4", decode_times=False)
        try:
            point = (
                ds["SST"]
                .sel(latitude=target_lat, longitude=target_lon, method="nearest")
                .isel(time=0, depth=0)
            )

            nearest_lat = float(
                ds["latitude"]
                .sel(latitude=target_lat, method="nearest")
                .values
            )
            nearest_lon = float(
                ds["longitude"]
                .sel(longitude=target_lon, method="nearest")
                .values
            )

            t_utc = parse_sst_time_from_filename(fp)
            rows.append(
                {
                    "file": fp.name,
                    "time_utc": t_utc,
                    "time_jst": t_utc + pd.Timedelta(hours=9),
                    "sst": float(point.values),
                    "nearest_lat": nearest_lat,
                    "nearest_lon": nearest_lon,
                }
            )
        except Exception as e:
            logger.warning("Skipping %s: %s", fp.name, e)
        finally:
            ds.close()

    if not rows:
        return pd.DataFrame(
            columns=["file", "time_utc", "time_jst", "sst", "nearest_lat", "nearest_lon"]
        )

    df = pd.DataFrame(rows).sort_values("time_utc").reset_index(drop=True)
    logger.info("SST point time series: %d records", len(df))
    return df


# ---------------------------------------------------------------------------
# Daily regional summary
# ---------------------------------------------------------------------------
def compute_daily_summary(
    root: Path,
    lat_min: float = 38.0,
    lat_max: float = 39.0,
    lon_min: float = 141.0,
    lon_max: float = 142.0,
) -> pd.DataFrame:
    """
    For each NetCDF file, compute regional SST statistics (mean, min,
    max, std) over the specified lat/lon bounding box, then aggregate
    to daily summaries.

    Returns DataFrame with columns:
        date, date_jst, mean_sst, min_sst, max_sst, std_sst, n_files
    """
    import xarray as xr

    files = list_sst_files(root)
    hourly_rows: List[Dict[str, Any]] = []

    for fp in files:
        ds = xr.open_dataset(str(fp), engine="netcdf4", decode_times=False)
        try:
            sst = ds["SST"].isel(time=0, depth=0)
            lat = ds["latitude"].values
            lon = ds["longitude"].values

            lat_mask = (lat >= lat_min) & (lat <= lat_max)
            lon_mask = (lon >= lon_min) & (lon <= lon_max)

            region = sst.values[np.ix_(lat_mask, lon_mask)].astype("float32")
            valid = region[np.isfinite(region)]

            if valid.size == 0:
                continue

            t_utc = parse_sst_time_from_filename(fp)
            hourly_rows.append(
                {
                    "file": fp.name,
                    "time_utc": t_utc,
                    "time_jst": t_utc + pd.Timedelta(hours=9),
                    "mean_sst": float(np.mean(valid)),
                    "min_sst": float(np.min(valid)),
                    "max_sst": float(np.max(valid)),
                    "std_sst": float(np.std(valid)),
                    "n_pixels": int(valid.size),
                }
            )
        except Exception as e:
            logger.warning("Skipping %s: %s", fp.name, e)
        finally:
            ds.close()

    if not hourly_rows:
        return pd.DataFrame(
            columns=["date", "date_jst", "mean_sst", "min_sst", "max_sst", "std_sst", "n_files"]
        )

    hourly = pd.DataFrame(hourly_rows)
    hourly["date_jst"] = hourly["time_jst"].dt.date

    daily = (
        hourly.groupby("date_jst")
        .agg(
            mean_sst=("mean_sst", "mean"),
            min_sst=("min_sst", "min"),
            max_sst=("max_sst", "max"),
            std_sst=("std_sst", "mean"),
            n_files=("file", "count"),
        )
        .reset_index()
    )

    logger.info("SST daily summary: %d days", len(daily))
    return daily


# ---------------------------------------------------------------------------
# Himawari .DAT raw file parsing (optional)
# ---------------------------------------------------------------------------
def parse_himawari_dat(
    dat_path: Path,
    target_lat: float,
    target_lon: float,
) -> Optional[Dict[str, Any]]:
    """
    Parse a raw Himawari .DAT file via satpy and extract brightness
    temperature at the target point.

    Returns None if satpy is not available or parsing fails.
    """
    try:
        from satpy import Scene
    except ImportError:
        logger.warning("satpy not installed – skipping raw Himawari file: %s", dat_path)
        return None

    try:
        scn = Scene(reader="ahi_hsd", filenames=[str(dat_path)])
        scn.load(["B14"])  # 11.2 µm thermal IR band for SST proxy

        data = scn["B14"]
        lats, lons = data.attrs.get("area").get_lonlats()

        # Find nearest pixel
        dist = (lats - target_lat) ** 2 + (lons - target_lon) ** 2
        idx = np.unravel_index(np.argmin(dist), dist.shape)

        bt = float(data.values[idx])

        # Parse time from filename: HS_H09_YYYYMMDD_HHMM_...
        m = re.search(r"(\d{8})_(\d{4})", dat_path.name)
        if not m:
            return None

        t_utc = pd.to_datetime(f"{m.group(1)} {m.group(2)}", format="%Y%m%d %H%M")

        return {
            "file": dat_path.name,
            "time_utc": t_utc,
            "time_jst": t_utc + pd.Timedelta(hours=9),
            "brightness_temp": bt,
            "nearest_lat": float(lats[idx]),
            "nearest_lon": float(lons[idx]),
        }
    except Exception as e:
        logger.warning("Failed to parse Himawari DAT %s: %s", dat_path.name, e)
        return None
