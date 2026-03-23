import os
import re
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


# =========================
# Paths
# =========================
TRAIN_META_PATH = "metadata_train.csv"
TRAIN_LABEL_PATH = "y_train.csv"
TRAIN_TXT_DIR = "fulltext_txt"

# Update these when the test set is released
TEST_META_PATH = "metadata_test.csv"
TEST_TXT_DIR = "fulltext_txt"


# =========================
# Utility functions
# =========================
PAPER_ID_PATTERN = r"^10\.1101[_/].+_v\d+$"


def read_txt_file(path, base_dir):
    """Read a txt file using either the original path or base_dir + basename."""
    if pd.isna(path) or str(path).strip() == "":
        return ""

    path = str(path).strip()
    candidates = [
        path,
        os.path.join(base_dir, os.path.basename(path))
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    return f.read()
            except UnicodeDecodeError:
                try:
                    with open(candidate, "r", encoding="latin-1") as f:
                        return f.read()
                except Exception:
                    return ""
            except Exception:
                return ""

    return ""


def safe_text(x):
    return "" if pd.isna(x) else str(x)


def safe_len(x):
    return 0 if pd.isna(x) else len(str(x))


def count_authors(x):
    """Rough author count from a raw author string."""
    if pd.isna(x) or str(x).strip() == "":
        return 0
    parts = re.split(r";|, and | and |,", str(x))
    parts = [p.strip() for p in parts if p.strip()]
    return len(parts)


def clean_metadata(df):
    """Drop unnamed columns created by malformed CSV rows."""
    df = df.copy()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
    return df


def filter_valid_paper_ids(df):
    """Keep rows with valid bioRxiv/medRxiv-style paper IDs."""
    df = df.copy()
    return df[df["paper_id"].astype(str).str.match(PAPER_ID_PATTERN, na=False)].copy()


def clean_labels(df):
    """Clean label file and keep binary outcomes only."""
    df = filter_valid_paper_ids(df)
    df["outcome"] = pd.to_numeric(df["outcome"], errors="coerce")
    df = df[df["outcome"].isin([0, 1])].copy()
    df["outcome"] = df["outcome"].astype(int)
    return df


def prepare_features(df, txt_base_dir, full_text_max_chars=30000):
    """Build model features from metadata and full text."""
    df = df.copy()

    required_cols = [
        "title", "abstract", "authors", "category",
        "server", "type", "license", "txt_path"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    for col in required_cols:
        df[col] = df[col].map(safe_text)

    df["full_text"] = df["txt_path"].map(lambda x: read_txt_file(x, txt_base_dir))
    df["full_text"] = df["full_text"].map(safe_text)
    df["full_text_short"] = df["full_text"].str[:full_text_max_chars]

    df["text_all"] = (
        df["title"] + " " +
        df["abstract"] + " " +
        df["full_text_short"]
    )

    df["title_len"] = df["title"].map(safe_len)
    df["abstract_len"] = df["abstract"].map(safe_len)
    df["full_text_len"] = df["full_text_short"].map(safe_len)
    df["n_authors_rough"] = df["authors"].map(count_authors)
    df["full_text_char_count"] = pd.to_numeric(
        df.get("full_text_char_count", 0), errors="coerce"
    ).fillna(0)
    df["version"] = pd.to_numeric(
        df.get("version", 0), errors="coerce"
    ).fillna(0)

    return df


# =========================
# Final feature columns
# =========================
TEXT_COL = "text_all"

NUM_COLS = [
    "title_len",
    "abstract_len",
    "full_text_len",
    "n_authors_rough",
    "full_text_char_count",
    "version"
]

CAT_COLS = [
    "category",
    "server",
    "type",
    "license"
]


# =========================
# Final model
# Best config from CV:
# - text = title + abstract + full text
# - max_features = 100000
# - ngram_range = (1, 2)
# - min_df = 2
# - C = 2.0
# =========================
final_preprocess = ColumnTransformer(
    transformers=[
        (
            "text",
            TfidfVectorizer(
                max_features=100000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            ),
            TEXT_COL
        ),
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("scaler", StandardScaler())
            ]),
            NUM_COLS
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            CAT_COLS
        )
    ],
    remainder="drop"
)

final_model = Pipeline([
    (
        "prep",
        final_preprocess
    ),
    (
        "clf",
        LogisticRegression(
            max_iter=2000,
            C=2.0,
            class_weight="balanced",
            solver="liblinear"
        )
    )
])


# =========================
# Train final model
# =========================
meta_train = pd.read_csv(TRAIN_META_PATH)
label_train = pd.read_csv(TRAIN_LABEL_PATH)

meta_train = clean_metadata(meta_train)
meta_train = filter_valid_paper_ids(meta_train)
label_train = clean_labels(label_train)

train_df = meta_train.merge(label_train, on="paper_id", how="inner")
train_df = prepare_features(train_df, txt_base_dir=TRAIN_TXT_DIR)

X_train = train_df[[TEXT_COL] + NUM_COLS + CAT_COLS].copy()
y_train = train_df["outcome"].copy()

print(f"Training rows: {len(train_df)}")
print(f"Positive rate: {y_train.mean():.4f}")
print(f"Empty full-text ratio: {(train_df['full_text'].str.len() == 0).mean():.4f}")

final_model.fit(X_train, y_train)
print("Final model fitted.")


# =========================
# Predict on test set
# Uncomment when test data is available
# =========================
meta_test = pd.read_csv(TEST_META_PATH)
meta_test = clean_metadata(meta_test)
meta_test = filter_valid_paper_ids(meta_test)

test_df = prepare_features(meta_test, txt_base_dir=TEST_TXT_DIR)
X_test = test_df[[TEXT_COL] + NUM_COLS + CAT_COLS].copy()

test_pred = final_model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
   "paper_id": test_df["paper_id"],
   "score": test_pred
})

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Saved submission.csv")