#!/usr/bin/env python3
"""Synthetic charging session generator.

Produces a CSV with columns:
- site_id (int)
- timestamp (ISO8601)
- sessions (float)  # aggregated sessions or kWh for that hour

Usage:
    python data/synthetic_generator.py
"""

import os

import numpy as np
import pandas as pd


def generate_synthetic_data(
    n_sites: int = 10,
    n_days: int = 90,
    freq: str = "H",
    seed: int = 42,
    out_path: str = "data/raw/synthetic_sessions.csv",
) -> pd.DataFrame:
    np.random.seed(seed)
    start = pd.Timestamp("2024-01-01 00:00:00")
    periods = int(n_days * 24)
    hours = pd.date_range(start, periods=periods, freq=freq)
    rows = []
    for site in range(1, n_sites + 1):
        # baseline demand depends on site type
        base = np.random.uniform(2, 15)
        # daily pattern: two peaks (morning commute, evening)
        daily = 3 * (np.sin((hours.hour - 7) / 24 * 2 * np.pi) + 1) + 2 * (
            np.sin((hours.hour - 18) / 24 * 2 * np.pi) + 1
        )
        # weekly modifier
        weekly = np.where(hours.dayofweek < 5, 1.0, 0.7)
        noise = np.random.normal(0, 1.0, size=periods)
        values = np.maximum(
            0, base * weekly + daily + noise + np.random.normal(0, 1, periods)
        )
        for ts, v in zip(hours, values):
            rows.append((site, ts.isoformat(), round(float(v), 3)))
    df = pd.DataFrame(rows, columns=["site_id", "timestamp", "sessions"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved synthetic data to {out_path}")
    return df


if __name__ == "__main__":
    generate_synthetic_data()
