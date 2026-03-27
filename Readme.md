# 📉 End-to-End ML Pipeline — Telecom Customer Churn Prediction

A complete, production style machine learning pipeline that combines **supervised classification** and **unsupervised clustering** to predict customer churn and segment customers for targeted marketing in the telecom domain.

---

## 📚 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Architecture Diagram](#architecture-diagram)
- [Technology Stack](#technology-stack)
- [Security](#security)
- [Limitations](#limitations)

---

## 🧭 Project Overview

A telecom company needs to proactively identify customers likely to cancel their subscription (churn) and understand which customer segments exist in its base for more effective marketing. Reacting to churn after the fact is costly, the goal is early, accurate prediction.

This project solves both problems in a single end-to-end pipeline:

| Objective | Approach |
|---|---|
| Predict which customers will churn | Binary classification (Supervised ML) |
| Group customers by behavior profile | K-Means Clustering (Unsupervised ML) |

**Dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  IBM Watson Analytics dataset containing ~7,000 telecom customers with 21 features including demographics, account info, and service usage.

**Key results demonstrated by the pipeline:**
- Multi-model comparison across Logistic Regression, Random Forest, Gradient Boosting, and XGBoost
- XGBoost selected as best model, tuned with GridSearchCV over a hyperparameter grid
- Evaluated with Accuracy, ROC-AUC, Confusion Matrix, and Feature Importance
- 3-cluster customer segmentation visualized via PCA reduction
- Full model persistence with `joblib` for downstream deployment

---

## 🏗️ Architecture

The pipeline is organized into six sequential phases, each building on the last with no data leakage between training and test sets.

### Phase 1 - Exploratory Data Analysis (EDA)
Baseline understanding of the dataset before any transformations. Includes target distribution analysis (churn ratio), boxplots of numeric features split by churn label, and stacked bar charts for categorical features. Highlights class imbalance (~26% churn) and key signals like contract type and tenure.

### Phase 2 - Data Preprocessing
Raw data cleaning steps applied before pipeline construction: `TotalCharges` coerced from string to numeric, `customerID` dropped as a non-informative identifier, and the target `Churn` label-encoded to binary (1/0). Rows with NaN in critical columns are removed.

### Phase 3 - ML Pipeline Construction
A `scikit-learn` `Pipeline` wraps all preprocessing and model steps to prevent leakage. Numeric features go through median imputation → standard scaling. Categorical features go through mode imputation → one-hot encoding. Four classifiers are evaluated via 5 fold cross-validation and compared on accuracy.

### Phase 4 - Hyperparameter Tuning
XGBoost, identified as the top-performing model, is tuned with `GridSearchCV` over `n_estimators`, `max_depth`, `learning_rate`, and `subsample`. Optimization target is ROC-AUC. Best estimator is retained as `best_model`.

### Phase 5 - Model Evaluation
The final model is evaluated on a held-out test set with full metrics: accuracy, ROC-AUC score, classification report (precision, recall, F1 per class), confusion matrix heatmap, ROC curve, and top-15 feature importance chart.

### Phase 6 - Customer Segmentation
The preprocessed feature matrix is reduced to 2 principal components via PCA and clustered using K-Means (K=3, selected via elbow method). Each cluster is profiled by churn rate and numeric feature means to support targeted marketing decisions.

---

## 🗺️ Architecture Diagram

```
Raw CSV (Kaggle)
      │
      ▼
┌─────────────────────────────┐
│   Phase 1: EDA              │
│  - Distribution analysis    │
│  - Feature-churn plots      │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Phase 2: Preprocessing    │
│  - Type coercion            │
│  - Target encoding          │
│  - Null removal             │
└────────────┬────────────────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
┌──────────┐  ┌─────────────────────────────┐
│ Numeric  │  │ Categorical                 │
│ Features │  │ Features                    │
│          │  │                             │
│ Impute   │  │ Impute (mode)               │
│ (median) │  │ → One-Hot Encode            │
│ → Scale  │  │                             │
└────┬─────┘  └──────────────┬──────────────┘
     │                       │
     └──────────┬────────────┘
                │ ColumnTransformer
                ▼
┌─────────────────────────────┐
│   Phase 3: Model Comparison │
│  - Logistic Regression      │
│  - Random Forest            │
│  - Gradient Boosting        │
│  - XGBoost ← best           │
│  (5-fold cross-validation)  │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Phase 4: Tuning           │
│  - GridSearchCV on XGBoost  │
│  - Optimized for ROC-AUC    │
└────────────┬────────────────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
┌──────────────┐  ┌───────────────────────┐
│  Phase 5:    │  │  Phase 6:             │
│  Evaluation  │  │  Segmentation         │
│              │  │                       │
│  Accuracy    │  │  PCA (2 components)   │
│  ROC-AUC     │  │  K-Means (K=3)        │
│  Confusion   │  │  Cluster profiling    │
│  Matrix      │  │  by churn rate        │
│  Feature Imp │  │                       │
└──────┬───────┘  └───────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│   Model Persistence         │
│   joblib → .pkl file        │
└─────────────────────────────┘
```


---

## 🛠️ Technology Stack

| Layer | Tool / Library | Purpose |
|---|---|---|
| **Data Ingestion** | `kagglehub` | Programmatic dataset download from Kaggle |
| **Data Manipulation** | `pandas`, `numpy` | Tabular data handling, feature engineering |
| **Preprocessing** | `scikit-learn` - `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `ColumnTransformer` | Null handling, scaling, encoding |
| **Pipeline** | `scikit-learn` - `Pipeline` | Leakage-free end-to-end transform + model chains |
| **Classification** | `scikit-learn` - `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier` | Baseline and ensemble classifiers |
| **Boosting** | `xgboost` - `XGBClassifier` | Best-performing gradient boosting model |
| **Tuning** | `scikit-learn` - `GridSearchCV` | Exhaustive hyperparameter search with cross-validation |
| **Dimensionality Reduction** | `scikit-learn` - `PCA` | 2-component projection for cluster visualization |
| **Clustering** | `scikit-learn` - `KMeans` | Unsupervised customer segmentation |
| **Evaluation** | `scikit-learn` - `accuracy_score`, `roc_auc_score`, `classification_report`, `confusion_matrix`, `roc_curve` | Full classification metrics suite |
| **Visualization** | `matplotlib`, `seaborn` | EDA plots, ROC curve, confusion matrix, feature importance |
| **Model Persistence** | `joblib` | Save and reload the trained pipeline as a `.pkl` file |
| **Environment** | Python 3.x, Jupyter Notebook | Development and execution environment |

---

## 🔐 Security

This project is a local, notebook-based ML pipeline with no live service endpoints, user authentication, or database connections. The following considerations apply for responsible use and any future deployment:

**Data Privacy**
- The dataset used is a publicly available, anonymized IBM/Watson synthetic dataset. No real customer PII is processed.
- If adapted to real telecom data, all customer identifiers must be removed or pseudonymized before ingestion. The pipeline already drops `customerID` by default.

**Credential Management**
- Kaggle API credentials (`kaggle.json`) are required for dataset download via `kagglehub`. These credentials must **never** be committed to version control.
- Add `kaggle.json` and any `.env` files to `.gitignore` before pushing to any repository.

**Model File Integrity**
- The saved `.pkl` file (`churn_prediction_pipeline.pkl`) is a serialized Python object. Only load `.pkl` files from trusted sources — malicious pickle files can execute arbitrary code on deserialization.
- If distributing the model, consider using `ONNX` or `mlflow` model registry formats with format-level validation.

**Dependency Security**
- Pin library versions in a `requirements.txt` to prevent supply-chain drift. Regularly audit with `pip audit` or `safety check`.

---

## ⚠️ Limitations

**Class Imbalance**
The dataset has approximately 74% non-churn vs. 26% churn. The pipeline does not apply SMOTE, class weighting, or threshold adjustment. Precision/recall on the minority class (churners) may be suboptimal for high-recall business needs. This is a known gap for future iteration.

**Hyperparameter Search Scope**
`GridSearchCV` is run over a fixed, manually defined parameter grid. This is computationally expensive for larger grids and does not guarantee a globally optimal configuration. Bayesian optimization (`Optuna`, `hyperopt`) would be more efficient for wider search spaces.

**No Drift Monitoring**
The pipeline has no mechanisms for detecting data drift or model degradation over time. In a production deployment, feature distribution shifts (e.g., changes in customer plan mix) would silently degrade predictions without alerting.

