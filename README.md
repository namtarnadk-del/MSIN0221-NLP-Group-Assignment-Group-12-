# Star rating vs. text sentiment in restaurant reviews

This repository supports the MSIN0221 group project on **star rating–text mismatch** in UK restaurant reviews. The main analysis lives in **`NLP Final Version 2.ipynb`**.

The notebook predicts **1–5 star ratings** from review text using:

- **TF-IDF** lexical features (unigrams/bigrams, sublinear TF), and  
- **RoBERTa sentiment probabilities** from a Hugging Face pipeline (`cardiffnlp/twitter-roberta-base-sentiment-latest`), used as dense features and for **zero-shot mismatch** labelling.

Supervised models (**Logistic Regression** and **LinearSVC**) are trained for both feature sets (four models in total), with **5-fold stratified `GridSearchCV`** over `C` and `class_weight` (macro F1). The notebook also includes **simple and threshold-based mismatch rules**, **error/lexical inspection**, and **external evaluation** on Yelp reviews without retraining.

## Data files

Place these CSV files in the **same directory** as the notebook (or update paths in the notebook):

| File | Role |
|------|------|
| `uk_restaurant_reviews.csv` | Primary dataset (Google Maps–style UK restaurant reviews). |
| `Yelp Restaurant Reviews.csv` | Optional external set; cells at the end score the **already trained** models for generalisation checks. |

Expected Yelp columns after rename (handled in the notebook): `Review Text` → review text, `Rating` → integer star rating.

## Environment

- **Python**: 3.10+ recommended (3.11 works with the listed packages).  
- **Hardware**: CPU is sufficient; a **GPU** speeds up RoBERTa feature extraction and any re-runs on large text batches.  
- **Disk / network**: First run downloads the RoBERTa weights via Hugging Face (cached for later runs).

## Setup

```bash
cd "/path/to/NLP Group 2"
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you use a **GPU**, install a CUDA-enabled build of PyTorch from the [official install guide](https://pytorch.org/get-started/locally/) instead of (or after) the generic `torch` line.

The notebook downloads **NLTK** resources when run:

- `stopwords`  
- `wordnet`  

Run all cells in order from the top so definitions (`preprocess_text`, `tfidf`, trained models, `results`, etc.) exist before **Yelp** cells.

## Reproducibility

- Stratified splits and several estimators use **`random_state=42`** where applicable.  
- RoBERTa outputs depend on library and model versions; small numerical differences are normal across environments.

## Repository layout (typical)

- `NLP Final Version 2.ipynb` — main pipeline (cleaning, EDA, features, tuning, models, mismatch analysis, Yelp evaluation).  
- `requirements.txt` — dependencies inferred from notebook imports.  
- `Group_Project_Report.md` / `.docx` — report artefacts (if present).  
- Earlier notebook variants (`Main Notebook.ipynb`, `NT_version.ipynb`) may exist for history only.

## Course context

Assignment brief and proposal PDFs in the folder describe marking criteria and the original project scope; this README only documents how to run the **final** notebook and its dependencies.
