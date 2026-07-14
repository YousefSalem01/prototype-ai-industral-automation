"""Fetch + assemble the REAL furnace/combustion dataset.

Source: UCI "Gas Turbine CO and NOx Emission Data Set" (dataset id 551) --
36,733 hourly sensor records from a real power-plant gas turbine in Turkey,
2011-2015. It is a genuine combustion process (the closest freely-downloadable,
login-free analog to a cement kiln) with the shape this project expects:
ambient conditions + operating variables + measured emission outputs.

Provenance / citation:
    Kaya, H., Tufekci, P., Uzun, E. (2019). "Predicting CO and NOx emissions
    from gas turbines." UCI Machine Learning Repository.
    https://archive.ics.uci.edu/dataset/551

This script downloads the zip (no Kaggle/login required), concatenates the five
yearly CSVs, and writes them to the configured dataset path. Column names are
kept exactly as published (AT, AP, AH, AFDP, GTEP, TIT, TAT, TEY, CDP, CO, NOX)
so that process_config.yaml -- the contract -- maps roles onto them.

Run:
    python -m src.fetch_real_data
"""

from __future__ import annotations

import io
import ssl
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .schema import load_config

_UCI_URL = (
    "https://archive.ics.uci.edu/static/public/551/"
    "gas+turbine+co+and+nox+emission+data+set.zip"
)


def download_and_assemble(dest: Path) -> pd.DataFrame:
    """Download the UCI zip and return the concatenated DataFrame.

    Args:
        dest: Path the combined CSV will be written to.

    Returns:
        The full combined DataFrame (also written to ``dest``).
    """
    ctx = ssl.create_default_context()
    req = urllib.request.Request(_UCI_URL, headers={"User-Agent": "Mozilla/5.0"})
    print(f"[fetch] downloading {_UCI_URL}")
    raw = urllib.request.urlopen(req, timeout=120, context=ctx).read()
    print(f"[fetch] {len(raw):,} bytes")

    zf = zipfile.ZipFile(io.BytesIO(raw))
    members = sorted(m for m in zf.namelist() if m.lower().endswith(".csv"))
    frames = [pd.read_csv(io.BytesIO(zf.read(m))) for m in members]
    df = pd.concat(frames, ignore_index=True)
    print(f"[fetch] assembled {len(df):,} rows x {df.shape[1]} cols "
          f"from {len(members)} yearly files")

    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"[fetch] wrote {dest}")
    return df


def main(config_path: str | None = None) -> Path:
    """CLI entry point: fetch the real dataset to the configured path."""
    config = load_config(config_path)
    out = config.data_path
    download_and_assemble(out)
    return out


if __name__ == "__main__":
    main()
