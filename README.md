# ml-hackathon-preprint-prediction
Preprint publication prediction using metadata, full text, and TF-IDF + logistic regression.
# Preprint Publication Prediction

A machine learning pipeline for predicting whether a research preprint will eventually be published in a top-tier journal using metadata and full-text features.

## Overview

This repository contains our final solution for a machine learning hackathon task on preprint outcome prediction.

The goal is to estimate the probability that a given research preprint will later be published in a top-tier journal. Our final pipeline combines:

- metadata features
- title and abstract text
- full-text features from `.txt` files
- TF-IDF text representation
- Logistic Regression for probabilistic prediction

## Method

Our final model uses:

- **Text input**: title + abstract + full text
- **Text representation**: TF-IDF
- **Classifier**: Logistic Regression
- **Additional features**:
  - title length
  - abstract length
  - full-text length
  - rough author count
  - full text character count
  - version
  - categorical metadata such as category, server, type, and license

### Final configuration

The final validated configuration is:

- `max_features=100000`
- `ngram_range=(1, 2)`
- `min_df=2`
- `max_df=0.95`
- `sublinear_tf=True`
- `LogisticRegression(C=2.0, class_weight="balanced", solver="liblinear")`

## Repository structure

```text
.
├── final_submission.py
├── README.md
├── requirements.txt
├── slides.pdf
└── slides.pptx
