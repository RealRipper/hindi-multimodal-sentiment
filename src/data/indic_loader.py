"""
IndicSentiment loader for Hindi binary sentiment.

Labels in the raw dataset: {None, 'Negative', 'Positive'}
None == neutral (no 'Neutral' string exists). Dropped per v1 decision.
Returned label encoding: 0 = Negative, 1 = Positive.
"""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

LABEL_MAP = {"Negative": 0, "Positive": 1}


def load_hindi_binary(
    seed: int = 42,
    val_size: float = 0.10,
    test_size: float = 0.10,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = load_dataset("ai4bharat/IndicSentiment", "translation-hi", split="test")
    df = raw.to_pandas()

    col_map = {}
    for col in df.columns:
        if "review" in col.lower() or "text" in col.lower() or "sentence" in col.lower():
            col_map[col] = "text"
        if "label" in col.lower() or "sentiment" in col.lower():
            col_map[col] = "label"
    df = df.rename(columns=col_map)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Could not identify text/label columns. Found: {list(df.columns)}"
        )

    total_before = len(df)
    df = df[["text", "label"]].copy()

    none_mask = df["label"].isna() | (df["label"] == "None") | (df["label"] == None)
    n_dropped = none_mask.sum()
    df = df[~none_mask].copy()

    if verbose:
        print(f"[indic_loader] Total rows loaded      : {total_before}")
        print(f"[indic_loader] None-label rows dropped: {n_dropped}")
        print(f"[indic_loader] Rows after filtering   : {len(df)}")

    unknown_mask = ~df["label"].isin(LABEL_MAP.keys())
    n_unknown = unknown_mask.sum()
    if n_unknown:
        logger.warning(
            "%d rows with unexpected labels dropped: %s",
            n_unknown,
            df.loc[unknown_mask, "label"].unique().tolist(),
        )
        df = df[~unknown_mask].copy()

    df["label"] = df["label"].map(LABEL_MAP)

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )
    val_fraction_of_remainder = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_fraction_of_remainder,
        random_state=seed,
        stratify=train_val_df["label"],
    )

    if verbose:
        for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            counts = split_df["label"].value_counts().to_dict()
            neg = counts.get(0, 0)
            pos = counts.get(1, 0)
            print(
                f"[indic_loader] {name:5s}: {len(split_df):4d} rows  "
                f"| Neg={neg}  Pos={pos}"
            )

    return train_df, val_df, test_df


if __name__ == "__main__":
    train, val, test = load_hindi_binary(seed=42, verbose=True)
