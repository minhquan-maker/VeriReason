"""Home Credit Default Risk: application table plus optional bureau / instalment aggregates.

Download (needs a Kaggle account and accepting the competition rules):
    kaggle competitions download -c home-credit-default-risk -p data/raw/home-credit
    unzip data/raw/home-credit/home-credit-default-risk.zip -d data/raw/home-credit
(or fetch only application_train.csv, bureau.csv, installments_payments.csv with `-f`.)

Aggregates are built per applicant (SK_ID_CURR) from that applicant's own records only. All
`DAYS_*` columns in the secondary tables are relative to the application date, so nothing after
the application is used. Applicants without records get zeros and a `HAS_*_HISTORY = 0` flag; the
flag sits in the same reason group as the aggregates it qualifies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DAYS_EMPLOYED_SENTINEL = 365243   # known placeholder for "not employed / pensioner"

_RAW_COLUMNS = [
    "SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
    "AMT_GOODS_PRICE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE", "CNT_CHILDREN", "CNT_FAM_MEMBERS", "DAYS_BIRTH", "DAYS_EMPLOYED",
    "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "REGION_RATING_CLIENT_W_CITY", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
    "DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE", "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE", "CODE_GENDER",
]


BUREAU_COLUMNS = ["BUREAU_OVERDUE_MAX", "BUREAU_OVERDUE_AMT", "BUREAU_DEBT_SUM",
                  "BUREAU_CREDIT_SUM", "BUREAU_ACTIVE_CNT", "HAS_BUREAU_HISTORY"]
INSTALMENT_COLUMNS = ["INST_LATE_RATIO", "INST_DAYS_LATE_MEAN", "INST_DAYS_LATE_MAX",
                      "HAS_INST_HISTORY"]
_BUREAU_USECOLS = ["SK_ID_CURR", "CREDIT_ACTIVE", "CREDIT_DAY_OVERDUE", "AMT_CREDIT_SUM",
                   "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE"]
_INST_USECOLS = ["SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"]
_CHUNK_ROWS = 2_000_000


def bureau_aggregates(path: Path, ids: pd.Series) -> pd.DataFrame:
    """Per-applicant aggregates of bureau.csv, indexed by SK_ID_CURR."""
    b = pd.read_csv(path, usecols=_BUREAU_USECOLS)
    b = b[b["SK_ID_CURR"].isin(ids)]
    b = b.assign(active=(b["CREDIT_ACTIVE"] == "Active").astype(int))
    g = b.groupby("SK_ID_CURR")
    agg = pd.DataFrame({
        "BUREAU_OVERDUE_MAX": g["CREDIT_DAY_OVERDUE"].max().clip(lower=0),
        "BUREAU_OVERDUE_AMT": g["AMT_CREDIT_SUM_OVERDUE"].sum(),
        "BUREAU_DEBT_SUM": g["AMT_CREDIT_SUM_DEBT"].sum(),
        "BUREAU_CREDIT_SUM": g["AMT_CREDIT_SUM"].sum(),
        "BUREAU_ACTIVE_CNT": g["active"].sum(),
    })
    agg["HAS_BUREAU_HISTORY"] = 1
    return agg


def instalment_aggregates(path: Path, ids: pd.Series) -> pd.DataFrame:
    """Per-applicant payment-behaviour aggregates of installments_payments.csv (read in chunks).

    days_late = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, clipped at 0 so early payments do not offset
    late ones. Instalments with no recorded payment date are ignored.
    """
    parts = []
    for chunk in pd.read_csv(path, usecols=_INST_USECOLS, chunksize=_CHUNK_ROWS):
        chunk = chunk[chunk["SK_ID_CURR"].isin(ids)].dropna(subset=["DAYS_ENTRY_PAYMENT"])
        late = (chunk["DAYS_ENTRY_PAYMENT"] - chunk["DAYS_INSTALMENT"]).clip(lower=0)
        part = pd.DataFrame({"id": chunk["SK_ID_CURR"], "late": late})
        part["is_late"] = (part["late"] > 0).astype(int)
        parts.append(part.groupby("id").agg(n=("late", "size"), n_late=("is_late", "sum"),
                                            sum_late=("late", "sum"), max_late=("late", "max")))
    if not parts:
        return pd.DataFrame(columns=INSTALMENT_COLUMNS)
    both = pd.concat(parts)
    g = both.groupby(level=0)
    tot = pd.DataFrame({"n": g["n"].sum(), "n_late": g["n_late"].sum(),
                        "sum_late": g["sum_late"].sum(), "max_late": g["max_late"].max()})
    agg = pd.DataFrame({
        "INST_LATE_RATIO": tot["n_late"] / tot["n"],
        "INST_DAYS_LATE_MEAN": tot["sum_late"] / tot["n"],
        "INST_DAYS_LATE_MAX": tot["max_late"],
    })
    agg["HAS_INST_HISTORY"] = 1
    agg.index.name = "SK_ID_CURR"
    return agg


def load(path: Path, max_rows: int | None = None, seed: int = 0,
         bureau_path: Path | None = None, installments_path: Path | None = None) -> pd.DataFrame:
    """Load applicants; add bureau / instalment aggregates when their paths are given."""
    for p in (path, bureau_path, installments_path):
        if p is not None and not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Download it from Kaggle (see module docstring).")
    df = pd.read_csv(path, usecols=_RAW_COLUMNS)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    for cols, fn, src in ((BUREAU_COLUMNS, bureau_aggregates, bureau_path),
                          (INSTALMENT_COLUMNS, instalment_aggregates, installments_path)):
        if src is None:
            continue
        agg = fn(src, df["SK_ID_CURR"])
        df = df.merge(agg, how="left", left_on="SK_ID_CURR", right_index=True)
        df[cols] = df[cols].fillna(0)

    out = pd.DataFrame({"applicant_id": "hc" + df.pop("SK_ID_CURR").astype(str)})
    days_emp = df.pop("DAYS_EMPLOYED").replace(DAYS_EMPLOYED_SENTINEL, np.nan)
    df["AGE_YEARS"] = -df.pop("DAYS_BIRTH") / 365.25
    df["YEARS_EMPLOYED"] = -days_emp / 365.25
    df["YEARS_ID_PUBLISH"] = -df.pop("DAYS_ID_PUBLISH") / 365.25
    df["YEARS_LAST_PHONE_CHANGE"] = -df.pop("DAYS_LAST_PHONE_CHANGE") / 365.25
    for c in ["NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
              "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "FLAG_OWN_CAR",
              "FLAG_OWN_REALTY", "CODE_GENDER"]:
        df[c] = df[c].fillna("missing").astype(str)
    return pd.concat([out, df.reset_index(drop=True)], axis=1)
