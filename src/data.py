"""FairHire AI — Phase 1 data pipeline.

Loads the UCI Adult Income dataset, defines the main task label and the three
sensitive attributes (gender, age bucket, ethnicity), cleans / encodes /
normalizes the features, builds stratified train/val/test splits, and exposes a
torch ``Dataset`` / ``DataLoader`` that yields ``(features, main_label,
sensitive_labels)``.

Design choices (see CLAUDE.md decisions log):
  * The three sensitive columns -- ``sex``, ``race`` and ``age`` -- are removed
    from the input features. The adversarial demo is only meaningful if the
    auditors must recover a sensitive attribute from *proxy* correlations in the
    latent ``z``, not from a copy of the attribute itself.
  * Preprocessing artifacts (scaler, one-hot categories) are fit on the TRAIN
    split only and reused for val/test, so there is no information leakage.
  * Everything is offline-friendly: raw files are cached under ``data/raw`` and
    the processed tensors under ``data/processed``.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

_UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult"
_RAW_FILES = {"adult.data": f"{_UCI_BASE}/adult.data",
              "adult.test": f"{_UCI_BASE}/adult.test"}

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

# Sensitive columns are stripped from the model input and kept only as labels.
SENSITIVE_COLUMNS = ["sex", "race", "age"]
TARGET_COLUMN = "income"

# Continuous and categorical feature columns actually fed to the encoder.
# 'education' is dropped (redundant with the ordinal 'education_num').
CONTINUOUS_FEATURES = [
    "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week",
]
CATEGORICAL_FEATURES = [
    "workclass", "marital_status", "occupation", "relationship",
    "native_country",
]

# Sensitive-label vocabularies (fixed order -> stable integer codes).
GENDER_CLASSES = ["Female", "Male"]
ETHNICITY_CLASSES = [
    "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other",
]
# Age buckets: young / mid / senior. Edges are (-inf, 30], (30, 50], (50, inf).
AGE_BUCKET_EDGES = [30, 50]
AGE_BUCKET_CLASSES = ["<=30", "31-50", ">50"]

SENSITIVE_KEYS = ("gender", "age", "ethnicity")
SEED = 42


# --------------------------------------------------------------------------- #
# Download + raw load
# --------------------------------------------------------------------------- #
def download_raw(force: bool = False) -> None:
    """Fetch adult.data / adult.test into ``data/raw`` (cached).

    Downloads to a temporary ``.part`` file and renames on success, so an
    interrupted download never leaves a truncated file that a later run would
    silently treat as complete.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in _RAW_FILES.items():
        dest = RAW_DIR / name
        if dest.exists() and not force:
            continue
        tmp = dest.with_suffix(dest.suffix + ".part")
        print(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)


def _read_raw_file(path: Path, is_test: bool) -> pd.DataFrame:
    # adult.test has a junk first line and trailing '.' on the income label.
    skiprows = 1 if is_test else 0
    df = pd.read_csv(
        path, header=None, names=COLUMNS, skiprows=skiprows,
        sep=",", skipinitialspace=True, na_values="?",
    )
    df[TARGET_COLUMN] = df[TARGET_COLUMN].str.rstrip(".")
    return df


def load_raw() -> pd.DataFrame:
    """Load both raw files, concatenate, and do row-level cleaning."""
    download_raw()
    frames = [
        _read_raw_file(RAW_DIR / "adult.data", is_test=False),
        _read_raw_file(RAW_DIR / "adult.test", is_test=True),
    ]
    df = pd.concat(frames, ignore_index=True)

    # Drop fully-empty rows (the test file can carry a stray blank line).
    df = df.dropna(how="all").reset_index(drop=True)

    # Missing categoricals ('?') become an explicit 'Unknown' category rather
    # than dropping rows -- preserves sample size and is honest about gaps.
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Unknown")

    # Income may carry stray whitespace; normalize to the two canonical labels.
    df[TARGET_COLUMN] = df[TARGET_COLUMN].str.strip()
    return df


# --------------------------------------------------------------------------- #
# Label encoding
# --------------------------------------------------------------------------- #
def _encode_target(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET_COLUMN] == ">50K").astype(np.int64).to_numpy()


def _encode_gender(df: pd.DataFrame) -> np.ndarray:
    mapping = {c: i for i, c in enumerate(GENDER_CLASSES)}
    return df["sex"].map(mapping).astype(np.int64).to_numpy()


def _encode_ethnicity(df: pd.DataFrame) -> np.ndarray:
    mapping = {c: i for i, c in enumerate(ETHNICITY_CLASSES)}
    return df["race"].map(mapping).astype(np.int64).to_numpy()


def _encode_age_bucket(df: pd.DataFrame) -> np.ndarray:
    bins = [-np.inf, *AGE_BUCKET_EDGES, np.inf]
    codes = pd.cut(df["age"], bins=bins, labels=False, right=True)
    return codes.astype(np.int64).to_numpy()


# --------------------------------------------------------------------------- #
# Feature matrix
# --------------------------------------------------------------------------- #
@dataclass
class Preprocessor:
    """Fits on train, transforms any split into a dense float32 matrix."""

    scaler: StandardScaler = field(default_factory=StandardScaler)
    categories: dict[str, list[str]] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        self.scaler.fit(df[CONTINUOUS_FEATURES].to_numpy(dtype=np.float64))
        self.categories = {
            col: sorted(df[col].astype(str).unique().tolist())
            for col in CATEGORICAL_FEATURES
        }
        self.feature_names = list(CONTINUOUS_FEATURES)
        for col in CATEGORICAL_FEATURES:
            self.feature_names += [f"{col}={v}" for v in self.categories[col]]
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cont = self.scaler.transform(
            df[CONTINUOUS_FEATURES].to_numpy(dtype=np.float64)
        )
        blocks = [cont]
        for col in CATEGORICAL_FEATURES:
            cats = self.categories[col]
            cat_index = {v: i for i, v in enumerate(cats)}
            col_vals = df[col].astype(str).to_numpy()
            onehot = np.zeros((len(df), len(cats)), dtype=np.float64)
            for row, val in enumerate(col_vals):
                idx = cat_index.get(val)
                if idx is not None:  # unseen category -> all-zero block
                    onehot[row, idx] = 1.0
            blocks.append(onehot)
        return np.concatenate(blocks, axis=1).astype(np.float32)


def _encode_all_labels(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "main": _encode_target(df),
        "gender": _encode_gender(df),
        "age": _encode_age_bucket(df),
        "ethnicity": _encode_ethnicity(df),
    }


# --------------------------------------------------------------------------- #
# Splits + build
# --------------------------------------------------------------------------- #
def _stratified_split(
    df: pd.DataFrame, seed: int = SEED,
    val_frac: float = 0.15, test_frac: float = 0.15,
) -> dict[str, pd.DataFrame]:
    """Stratify on income so each split has the same positive rate."""
    rng = np.random.default_rng(seed)
    parts: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    for _, group in df.groupby(TARGET_COLUMN):
        idx = rng.permutation(len(group))
        n = len(group)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        g = group.iloc[idx]
        parts["test"].append(g.iloc[:n_test])
        parts["val"].append(g.iloc[n_test:n_test + n_val])
        parts["train"].append(g.iloc[n_test + n_val:])
    out = {}
    for split, chunks in parts.items():
        merged = pd.concat(chunks, ignore_index=True)
        out[split] = merged.sample(frac=1.0, random_state=seed).reset_index(
            drop=True
        )
    return out


def build_processed(force: bool = False) -> dict[str, np.ndarray]:
    """Full pipeline: load -> split -> fit/transform -> save .npz. Returns the
    in-memory bundle as well."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    bundle_path = PROCESSED_DIR / "adult.npz"
    if bundle_path.exists() and not force:
        return dict(np.load(bundle_path, allow_pickle=True))

    df = load_raw()
    splits = _stratified_split(df)

    pre = Preprocessor().fit(splits["train"])

    bundle: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        sdf = splits[split]
        bundle[f"X_{split}"] = pre.transform(sdf)
        labels = _encode_all_labels(sdf)
        for key, arr in labels.items():
            bundle[f"{key}_{split}"] = arr

    bundle["feature_names"] = np.array(pre.feature_names, dtype=object)
    bundle["continuous_features"] = np.array(CONTINUOUS_FEATURES, dtype=object)
    bundle["gender_classes"] = np.array(GENDER_CLASSES, dtype=object)
    bundle["age_classes"] = np.array(AGE_BUCKET_CLASSES, dtype=object)
    bundle["ethnicity_classes"] = np.array(ETHNICITY_CLASSES, dtype=object)
    bundle["scaler_mean"] = pre.scaler.mean_.astype(np.float64)
    bundle["scaler_scale"] = pre.scaler.scale_.astype(np.float64)

    np.savez(bundle_path, **bundle)
    print(f"Saved processed bundle -> {bundle_path}")
    return bundle


# --------------------------------------------------------------------------- #
# torch Dataset / DataLoader
# --------------------------------------------------------------------------- #
class FairHireDataset(Dataset):
    """Yields ``(features, main_label, sensitive_labels)`` where
    ``sensitive_labels`` is a dict with keys gender / age / ethnicity."""

    def __init__(self, bundle: dict[str, np.ndarray], split: str):
        self.X = torch.as_tensor(bundle[f"X_{split}"], dtype=torch.float32)
        self.main = torch.as_tensor(bundle[f"main_{split}"], dtype=torch.long)
        self.sensitive = {
            k: torch.as_tensor(bundle[f"{k}_{split}"], dtype=torch.long)
            for k in SENSITIVE_KEYS
        }
        self.n_features = self.X.shape[1]

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int):
        sens = {k: v[i] for k, v in self.sensitive.items()}
        return self.X[i], self.main[i], sens


def make_dataloaders(
    batch_size: int = 256, force: bool = False, **loader_kwargs,
) -> tuple[dict[str, DataLoader], dict[str, np.ndarray]]:
    """Returns ({train,val,test: DataLoader}, processed_bundle)."""
    bundle = build_processed(force=force)
    loaders = {}
    for split in ("train", "val", "test"):
        ds = FairHireDataset(bundle, split)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            drop_last=False, **loader_kwargs,
        )
    return loaders, bundle


if __name__ == "__main__":
    build_processed(force=True)
