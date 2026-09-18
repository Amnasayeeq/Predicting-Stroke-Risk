
# Predicting Stroke Risk: An End-to-End Machine Learning Pipeline


## Overview

Stroke is one of the leading causes of death worldwide, and much of that risk is preventable with early intervention. This project builds an end-to-end machine learning pipeline that flags patients at elevated stroke risk from routine demographic and clinical data, so a health provider can prioritise limited preventive-care resources (check-ups, lifestyle counselling, monitoring) toward the patients who need them most.

The notebook covers the full pipeline: problem framing, exploratory data analysis, preprocessing and feature engineering, model training and tuning, evaluation on a held-out test set, and an explainability/deployment discussion.

## Dataset

[Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) — fedesoriano, Kaggle.

- 5,110 patient records, 12 columns (11 features + `stroke` target)
- Highly imbalanced: only ~4.9% of records are positive (249 of 5,110)
- `bmi` has ~201 missing values (~3.9%)

| Column | Description |
|---|---|
| `gender` | Male / Female / Other |
| `age` | Patient age |
| `hypertension` | 0 / 1 |
| `heart_disease` | 0 / 1 |
| `ever_married` | Yes / No |
| `work_type` | Private, Self-employed, Govt_job, children, Never_worked |
| `Residence_type` | Urban / Rural |
| `avg_glucose_level` | Average blood glucose level |
| `bmi` | Body mass index |
| `smoking_status` | formerly smoked, never smoked, smokes, Unknown |
| `stroke` | Target — 1 if the patient had a stroke, else 0 |

## Methodology

**Preprocessing**
- Dropped the non-predictive `id` column
- Median imputation for missing `bmi`
- Engineered `age_group` (clinically meaningful age bands) and `glucose_risk` (binary flag for glucose above the diabetic threshold of 140)
- One-hot encoding for categorical features, standard scaling for numeric features
- Stratified 80/20 train/test split to preserve the ~4.9% positive rate in both sets
- All steps encapsulated in a single `scikit-learn` `Pipeline` + `ColumnTransformer` to prevent data leakage

**Handling class imbalance**
Two strategies were compared via stratified 5-fold cross-validation on F1-score: `class_weight='balanced'` and SMOTE oversampling (fit only within CV training folds). The higher-scoring approach was carried forward per model.

**Models compared**
- Logistic Regression — interpretable linear baseline
- Random Forest — captures non-linear feature interactions
- Gradient Boosting — typically strong on tabular data, less interpretable

**Explainability**
Random Forest feature importances plus SHAP values for patient-level explanations of individual predictions.

## Results (held-out test set)

| Model | Recall (stroke) | Precision (stroke) | F1 (stroke) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.80 | 0.125 | 0.216 | 0.837 | 0.244 |
| **Random Forest** | 0.72 | 0.198 | **0.310** | 0.838 | 0.236 |
| Gradient Boosting | 0.08 | 0.444 | 0.136 | 0.814 | 0.192 |

Accuracy is not used as the headline metric — with a ~95/5 class split, a model that always predicts "no stroke" would already score ~95% while being clinically useless. **Random Forest** was selected as the final model for its best balance of precision and recall (highest F1), while Logistic Regression offers the highest recall for use cases that prioritise catching every at-risk patient over minimising false alarms.

**Top predictive features:** age, average glucose level, senior age band, BMI, and adult age band — consistent with established clinical risk factors.

## Key findings & recommendations

- A risk-tiered output (e.g. LOW / MEDIUM / HIGH probability bands) is more clinically useful than a hard classification, since it lets clinicians set their own risk-tolerance threshold.
- The model should be deployed as a **decision-support signal alongside clinical judgement**, not a standalone diagnostic tool — the small number of positive cases (249) means recall estimates carry real uncertainty and should be validated on a larger, prospective sample before wider use.
- SMOTE's synthetic points can occasionally produce clinically implausible combinations (e.g. a child with a senior's risk profile); class weighting was preferred for the final Random Forest model on this dataset.

## Repository structure

```
├── README.md
├── stroke_prediction.ipynb       # Full analysis notebook
├── stroke_prediction.html        # Rendered HTML export (submission format)
├── healthcare-dataset-stroke-data.csv
└── requirements.txt
```

## Setup & usage

```bash
git clone https://github.com/Amnasayeeq/stroke-prediction.git
cd stroke-prediction
pip install -r requirements.txt
jupyter notebook stroke_prediction.ipynb
```

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
shap
```

## Limitations

- Small positive class (249 stroke cases) limits confidence in recall estimates
- No temporal information (e.g. duration of hypertension) to model risk trajectory over time
- Not externally validated on a different hospital or population

