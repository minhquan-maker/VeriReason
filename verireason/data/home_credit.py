"""Home Credit Default Risk — main application table only (pilot scope).

Download (needs a Kaggle account and accepting the competition rules):
    kaggle competitions download -c home-credit-default-risk -f application_train.csv \
        -p data/raw/home-credit
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


def load(path: Path, max_rows: int | None = None, seed: int = 0) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download application_train.csv from Kaggle (see module docstring).")
    df = pd.read_csv(path, usecols=_RAW_COLUMNS)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

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
    return pd.concat([out, df], axis=1)
