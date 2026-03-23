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

# ============================================================
# 1. SETUP PATHS
# ============================================================
# Update these paths to match your local machine directory
TRAIN_BASE_DIR = "preprint_train"
TEST_BASE_DIR = "preprint_test"

TRAIN_META_PATH = os.path.join(TRAIN_BASE_DIR, "metadata_train.csv")
TRAIN_LABEL_PATH = os.path.join(TRAIN_BASE_DIR, "y_train.csv")
TRAIN_TXT_DIR = os.path.join(TRAIN_BASE_DIR, "fulltext_txt")

TEST_META_PATH = os.path.join(TEST_BASE_DIR, "metadata_test.csv")
TEST_TXT_DIR = os.path.join(TEST_BASE_DIR, "fulltext_txt")

# Destination for the final submission file on Desktop
FINAL_OUTPUT_PATH = "/Users/huangshiqi/Desktop/final_submission_ready.csv"

# ============================================================
# 2. UTILITY FUNCTIONS
# ============================================================
def read_txt_file(paper_id, base_dir):
    """Reads the first 5000 characters of the txt file for a given paper_id."""
    path = os.path.join(base_dir, f"{paper_id}.txt")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read(5000) 
        except Exception:
            return ""
    return ""

def clean_text(text):
    """Removes noise tags like AO_SCPLOW and standardizes text."""
    if not isinstance(text, str): return ""
    # Remove uppercase noise tags (common in PDF-to-TXT conversions)
    text = re.sub(r'[A-Z]{2,}_[A-Z]{2,}', '', text) 
    return text

def prepare_features(df, txt_base_dir):
    """Processes metadata and full text to create a combined feature set."""
    df = df.copy()
    
    # Load text content from files
    print(f"Reading text files from {txt_base_dir}...")
    df["full_text_snippet"] = df["paper_id"].apply(lambda x: read_txt_file(x, txt_base_dir))
    
    # Combine title, abstract, and text snippet into one column for NLP
    df["combined_text"] = (
        df["title"].fillna("") + " " + 
        df["abstract"].fillna("") + " " + 
        df["full_text_snippet"].apply(clean_text)
    )
    
    # Extract numerical features
    df["title_len"] = df["title"].str.len().fillna(0)
    df["abstract_len"] = df["abstract"].str.len().fillna(0)
    df["num_authors"] = df["authors"].str.count('[,;]').fillna(0) + 1
    
    return df

# ============================================================
# 3. DEFINE MACHINE LEARNING PIPELINE
# ============================================================
TEXT_COL = "combined_text"
NUM_COLS = ["title_len", "abstract_len", "num_authors"]
CAT_COLS = ["category", "server", "license"]

# Preprocessing for different data types
preprocessor = ColumnTransformer(
    transformers=[
        # NLP: Convert text to TF-IDF vectors
        ("text", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words='english'), TEXT_COL),
        # Numerical: Fill missing values and scale
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), NUM_COLS),
        # Categorical: Fill missing values and One-Hot Encode
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), CAT_COLS)
    ]
)

# Final Pipeline with Logistic Regression
final_pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", solver="liblinear"))
])

# ============================================================
# 4. TRAINING PHASE
# ============================================================
print("Step 1: Loading Training Data...")
meta_train = pd.read_csv(TRAIN_META_PATH)
y_train_df = pd.read_csv(TRAIN_LABEL_PATH)

# Merge metadata and labels on paper_id
train_df = pd.merge(meta_train, y_train_df, on="paper_id")
train_df = prepare_features(train_df, TRAIN_TXT_DIR)

# Convert labels to numeric and drop any rows with missing targets
train_df["outcome"] = pd.to_numeric(train_df["outcome"], errors="coerce")
train_df = train_df.dropna(subset=["outcome"])
y_train = train_df["outcome"].astype(int)

print("Step 2: Training Model...")
final_pipeline.fit(train_df, y_train)
print("Model training complete.")

# ============================================================
# 5. PREDICTION PHASE
# ============================================================
print("Step 3: Loading Test Data...")
meta_test = pd.read_csv(TEST_META_PATH)

# Prepare features for the test set
test_df = prepare_features(meta_test, TEST_TXT_DIR)

print("Step 4: Generating Probabilistic Predictions...")
# Generate probabilities for outcome = 1
test_probs = final_pipeline.predict_proba(test_df)[:, 1]

# Create submission DataFrame with required column names
submission = pd.DataFrame({
    "paper_id": test_df["paper_id"],
    "prediction": test_probs  # Column name must be 'prediction' per hackathon rules
})

# Save to Desktop
submission.to_csv(FINAL_OUTPUT_PATH, index=False)

print(f"\n✅ SUCCESS! File saved as: {FINAL_OUTPUT_PATH}")
print(submission.head())
