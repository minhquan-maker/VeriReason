"""Statlog German Credit (UCI). 1000 applicants, 20 attributes, label 1=good / 2=bad."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# Column order of german.data (UCI documentation, attributes 1-20 then the class).
UCI_ORDER = [
    "checking_status", "duration_months", "credit_history", "purpose", "credit_amount",
    "savings", "employment_since", "installment_rate", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age", "other_installment_plans", "housing",
    "existing_credits", "job", "people_liable", "telephone", "foreign_worker", "class",
]


def download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path


def load(path: Path, url: str | None = None) -> pd.DataFrame:
    """Return a dataframe with the UCI columns plus `default` (1 = bad credit risk)."""
    if not path.exists():
        if not url:
            raise FileNotFoundError(f"{path} not found and no download url configured")
        download(url, path)
    df = pd.read_csv(path, sep=r"\s+", header=None, names=UCI_ORDER, dtype=str)
    if df.shape[1] != len(UCI_ORDER) or df.isna().any().any():
        raise ValueError(f"{path} does not look like UCI german.data (21 space-separated columns)")
    numeric = ["duration_months", "credit_amount", "installment_rate", "residence_since", "age",
               "existing_credits", "people_liable"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c])
    df["default"] = (df.pop("class").astype(int) == 2).astype(int)
    df.insert(0, "applicant_id", [f"de{i:04d}" for i in range(len(df))])
    return df


def synthesize(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """SYNTHETIC rows in the German Credit format, for offline smoke tests only.

    Values are random valid codes; the label follows a planted logistic rule (checking account,
    credit history, duration, savings) so that trained models have real structure. Never report
    results computed on this data.
    """
    rng = np.random.default_rng(seed)

    def cat(prefix: str, codes: list[int], p=None):
        return [f"{prefix}{c}" for c in rng.choice(codes, size=n, p=p)]

    df = pd.DataFrame({
        "checking_status": cat("A1", [1, 2, 3, 4], [0.27, 0.27, 0.06, 0.40]),
        "duration_months": rng.integers(4, 72, n),
        "credit_history": cat("A3", [0, 1, 2, 3, 4], [0.04, 0.05, 0.53, 0.09, 0.29]),
        "purpose": cat("A4", [0, 1, 2, 3, 4, 5, 6, 8, 9, 10]),
        "credit_amount": rng.lognormal(7.8, 0.75, n).round().astype(int),
        "savings": cat("A6", [1, 2, 3, 4, 5], [0.60, 0.10, 0.06, 0.05, 0.19]),
        "employment_since": cat("A7", [1, 2, 3, 4, 5]),
        "installment_rate": rng.integers(1, 5, n),
        "personal_status_sex": cat("A9", [1, 2, 3, 4]),
        "other_debtors": cat("A10", [1, 2, 3], [0.9, 0.04, 0.06]),
        "residence_since": rng.integers(1, 5, n),
        "property": cat("A12", [1, 2, 3, 4]),
        "age": rng.integers(19, 75, n),
        "other_installment_plans": cat("A14", [1, 2, 3], [0.14, 0.05, 0.81]),
        "housing": cat("A15", [1, 2, 3], [0.18, 0.71, 0.11]),
        "existing_credits": rng.integers(1, 5, n),
        "job": cat("A17", [1, 2, 3, 4], [0.02, 0.2, 0.63, 0.15]),
        "people_liable": rng.integers(1, 3, n),
        "telephone": cat("A19", [1, 2]),
        "foreign_worker": cat("A20", [1, 2], [0.96, 0.04]),
    })
    logit = (-1.2
             + 1.1 * (df.checking_status == "A11") + 0.6 * (df.checking_status == "A12")
             - 1.0 * (df.checking_status == "A14")
             + 0.9 * df.credit_history.isin(["A30", "A31"]) - 0.6 * (df.credit_history == "A34")
             + 0.03 * (df.duration_months - 20)
             + 0.5 * (df.savings == "A61") - 0.5 * df.savings.isin(["A63", "A64"]))
    df["default"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df.insert(0, "applicant_id", [f"syn{i:04d}" for i in range(n)])
    return df
