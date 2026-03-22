# 🎗️ Breast Cancer Tumor Classification

> A binary classification system that predicts whether a breast tumor is **Malignant** (cancerous) or **Benign** (non-cancerous) using an ensemble of machine learning models trained on digitized medical imaging data.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Architecture Diagram](#-architecture-diagram)
- [Data Flow](#-data-flow)
- [Technology Stack](#-technology-stack)
- [Security](#-security)
- [Limitations](#-limitations)

---

## 📌 Project Overview

Breast cancer is one of the most diagnosed cancers globally, with early detection being the single most critical factor in improving survival rates. This capstone project builds a robust **binary classification pipeline** to distinguish malignant tumors from benign ones using clinical imaging features.

### What This Project Does — Step by Step

**Step 1 — Load and Explore the Dataset**

The pipeline begins by loading the **Breast Cancer Wisconsin Dataset** directly from `sklearn.datasets`. This dataset contains 569 patient samples, each described by 30 numerical features derived from digitized images of fine needle aspirate (FNA) biopsies of breast masses. Features capture characteristics of cell nuclei such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension — each computed as the mean, standard error, and worst (largest) value across the image. Initial exploration includes inspecting class distributions, feature dtypes, summary statistics, and checking for null values.

**Step 2 — Data Cleaning and Preprocessing**

Raw data is validated and cleaned before any modeling begins. This includes verifying that no missing or duplicate values exist, confirming all features are numeric, and removing any corrupted rows. Features are then standardized using `StandardScaler` to ensure no single feature dominates due to scale differences — critical for distance-based and regularized models. The dataset is split into training (80%) and test (20%) sets using stratified sampling to preserve the original class ratio across both splits.

**Step 3 — Exploratory Data Analysis (EDA)**

EDA uncovers patterns and relationships within the data before any model training. This step produces:
- **Class distribution plots** to visualize the Malignant vs. Benign ratio
- **Feature distribution histograms** to inspect spread and skewness per class
- **Correlation heatmaps** to identify redundant or highly collinear features
- **Boxplots** comparing feature values across the two target classes
- **Pairplots** for the most discriminative features to visually assess separability

Key insight: features like `mean radius`, `mean perimeter`, and `worst concave points` show strong separation between classes, making them highly predictive.

**Step 4 — Feature Engineering**

Based on EDA findings, feature engineering refines the input space to improve model performance:
- **Feature selection** by removing features with near-zero variance or very high mutual correlation (threshold > 0.95), reducing multicollinearity
- **Principal Component Analysis (PCA)** applied for 2D visualization of class separability — not used in final model inputs to preserve interpretability
- **Feature importance ranking** using a preliminary Random Forest to identify the top contributing features

**Step 5 — Train Multiple ML Models**

Eight classifiers are trained on the same preprocessed training set to enable fair comparison:

| # | Model | Type | Key Characteristic |
|---|---|---|---|
| 1 | `LogisticRegression` | Linear | Probabilistic baseline; fast and interpretable |
| 2 | `DecisionTreeClassifier` | Tree-based | High interpretability; prone to overfitting |
| 3 | `RandomForestClassifier` | Ensemble (Bagging) | Reduces variance via averaging many trees |
| 4 | `GradientBoostingClassifier` | Ensemble (Boosting) | Sequentially corrects errors; high accuracy |
| 5 | `AdaBoostClassifier` | Ensemble (Boosting) | Focuses on hard-to-classify samples |
| 6 | `SVC` | Kernel-based | Effective in high-dimensional space |
| 7 | `KNeighborsClassifier` | Instance-based | Non-parametric; sensitive to feature scale |
| 8 | `GaussianNB` | Probabilistic | Fast; assumes feature independence |

All models use default hyperparameters in the first pass, with the top performers subsequently tuned via `GridSearchCV` with 5-fold cross-validation.

**Step 6 — Model Evaluation and Comparison**

Each model is evaluated on the held-out test set using a comprehensive set of metrics:
- **Accuracy** — overall correct predictions
- **Precision** — of all predicted malignant, how many truly are
- **Recall (Sensitivity)** — of all actual malignant cases, how many were caught *(prioritized, as false negatives are dangerous in medical contexts)*
- **F1-Score** — harmonic mean of precision and recall
- **ROC-AUC Score** — area under the receiver operating characteristic curve
- **Confusion Matrix** — breakdown of TP, TN, FP, FN per model

Models are ranked and compared side-by-side in a summary table and ROC curve overlay plot.

**Step 7 — Save the Best Model**

The best-performing model (based on F1-Score and Recall) is serialized to disk using `joblib` for future inference without retraining. The saved artifact includes the trained model and the fitted `StandardScaler` to ensure consistent preprocessing at prediction time.

---

## 🏗️ Architecture

The system is designed as a **sequential ML pipeline** with clearly separated stages:

```
Raw Dataset
    │
    ▼
Data Validation & Cleaning
    │
    ▼
Preprocessing (Scaling + Train/Test Split)
    │
    ▼
EDA & Feature Engineering
    │
    ├──────────────────────────────────────────────────┐
    ▼                                                  ▼
Model Training Layer                          Feature Insight Layer
(8 classifiers in parallel)                   (PCA, Correlation, Importance)
    │
    ▼
Evaluation & Comparison Engine
    │
    ▼
Best Model Selection
    │
    ▼
Model Serialization (joblib)
    │
    ▼
Saved Model + Scaler (ready for inference)
```

Each stage is modular — preprocessing, training, and evaluation can be run independently, making the pipeline easy to extend with new models or datasets.

---

## 🗺️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                   │
│                                                                     │
│   sklearn.datasets ──► Breast Cancer Wisconsin Dataset (569 x 30)  │
│                                  │                                  │
│                         [Validation & Cleaning]                     │
└──────────────────────────────────┼──────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────┐
│                     PREPROCESSING LAYER                             │
│                                                                     │
│   ┌──────────────┐    ┌──────────────┐    ┌─────────────────────┐  │
│   │ StandardScaler│   │ Train/Test   │    │  Feature Selection  │  │
│   │  (fit on     │──► │  Split (80/20│──► │  + PCA (EDA only)  │  │
│   │  train only) │    │  stratified) │    │                     │  │
│   └──────────────┘    └──────────────┘    └─────────────────────┘  │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────┐
│                      MODEL TRAINING LAYER                           │
│                                                                     │
│   ┌────────────┐  ┌────────────┐  ┌──────────────┐  ┌──────────┐  │
│   │  Logistic  │  │  Decision  │  │    Random    │  │ Gradient │  │
│   │ Regression │  │    Tree    │  │    Forest    │  │ Boosting │  │
│   └────────────┘  └────────────┘  └──────────────┘  └──────────┘  │
│                                                                     │
│   ┌────────────┐  ┌────────────┐  ┌──────────────┐  ┌──────────┐  │
│   │  AdaBoost  │  │    SVC     │  │     KNN      │  │ Gaussian │  │
│   │            │  │            │  │              │  │    NB    │  │
│   └────────────┘  └────────────┘  └──────────────┘  └──────────┘  │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────┐
│                     EVALUATION LAYER                                │
│                                                                     │
│   Accuracy │ Precision │ Recall │ F1 │ ROC-AUC │ Confusion Matrix  │
│                                                                     │
│                    ┌─────────────────┐                             │
│                    │  Best Model     │                             │
│                    │  Selection      │                             │
│                    └────────┬────────┘                             │
└─────────────────────────────┼───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                      PERSISTENCE LAYER                              │
│                                                                     │
│        best_model.pkl  ──────  scaler.pkl                          │
│         (joblib)                (joblib)                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow

```
1. SOURCE
   └── sklearn.datasets.load_breast_cancer()
       └── 569 samples × 30 features + 1 binary target label

2. VALIDATION
   └── Check for nulls, duplicates, dtype consistency
       └── Output: Clean DataFrame (no transformations yet)

3. PREPROCESSING
   ├── StandardScaler.fit(X_train) → StandardScaler.transform(X_train, X_test)
   ├── train_test_split(stratify=y, test_size=0.2, random_state=42)
   └── Output: X_train_scaled, X_test_scaled, y_train, y_test

4. EDA & FEATURE ENGINEERING
   ├── Correlation matrix → Drop features with correlation > 0.95
   ├── RandomForest feature importance → Rank features
   ├── PCA (2 components) → 2D scatter plot only (not used in model input)
   └── Output: Final feature set for training

5. MODEL TRAINING
   ├── 8 models trained independently on (X_train_scaled, y_train)
   ├── 5-Fold Cross Validation on training set
   └── Output: 8 fitted model objects

6. EVALUATION
   ├── Predictions: model.predict(X_test_scaled)
   ├── Probabilities: model.predict_proba(X_test_scaled) [where available]
   ├── Metrics computed per model
   └── Output: Comparison DataFrame + plots

7. PERSISTENCE
   ├── joblib.dump(best_model, 'models/best_model.pkl')
   └── joblib.dump(scaler, 'models/scaler.pkl')

8. INFERENCE (future use)
   ├── scaler = joblib.load('models/scaler.pkl')
   ├── model = joblib.load('models/best_model.pkl')
   ├── X_new_scaled = scaler.transform(X_new)
   └── prediction = model.predict(X_new_scaled)  →  0 (Benign) / 1 (Malignant)
```

---

## 🛠️ Technology Stack

| Category | Library / Tool | Version | Purpose |
|---|---|---|---|
| **Language** | Python | 3.9+ | Core programming language |
| **Data Handling** | Pandas | 2.x | DataFrame manipulation and analysis |
| **Numerical Computing** | NumPy | 1.x | Array operations and math utilities |
| **Machine Learning** | scikit-learn | 1.x | Models, preprocessing, evaluation, pipelines |
| **Visualization** | Matplotlib | 3.x | Plots, heatmaps, ROC curves |
| **Visualization** | Seaborn | 0.13.x | Statistical data visualization |
| **Model Persistence** | joblib | 1.x | Serializing and loading trained models |
| **Notebook Environment** | Jupyter Notebook | — | Interactive development and EDA |

### scikit-learn Modules Used

```python
# Data
from sklearn.datasets import load_breast_cancer

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report

# Dimensionality Reduction
from sklearn.decomposition import PCA
```

---

## 📊 Results

| # | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | CV Score |
|---|---|---|---|---|---|---|---|
| 1 | Logistic Regression | 0.9737 | 0.9859 | 0.9722 | 0.9790 | 0.9950 | 0.9824 |
| 2 | Support Vector Machine | 0.9737 | 0.9859 | 0.9722 | 0.9790 | 0.9950 | 0.9714 |
| 3 | Gradient Boosting | 0.9561 | 0.9467 | 0.9861 | 0.9660 | 0.9878 | 0.9582 |
| 4 | AdaBoost | 0.9561 | 0.9467 | 0.9861 | 0.9660 | 0.9815 | 0.9692 |
| 5 | K-Nearest Neighbors | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9901 | 0.9648 |
| 6 | Random Forest | 0.9474 | 0.9583 | 0.9583 | 0.9583 | 0.9932 | 0.9604 |
| 7 | Naive Bayes | 0.9298 | 0.9444 | 0.9444 | 0.9444 | 0.9838 | 0.9319 |
| 8 | Decision Tree | 0.9123 | 0.9429 | 0.9167 | 0.9296 | 0.9107 | 0.9319 |
---

## ⚠️ Limitations

**1. Feature Origin**
All 30 features are derived from a single FNA biopsy image per patient. Real-world diagnosis incorporates multiple imaging modalities (mammography, ultrasound, MRI), clinical history, and pathology — none of which are represented here.


**2. Class Imbalance**
The dataset is mildly imbalanced (~63% Benign, ~37% Malignant). While stratified splits help, models may still be slightly biased toward predicting Benign. Techniques like SMOTE or class weighting were not applied in this baseline pipeline.

**3. No Hyperparameter Optimization at Scale**
Hyperparameter tuning via `GridSearchCV` is applied only to the top-performing models. A full Bayesian optimization or `RandomizedSearchCV` sweep across all models was out of scope for this capstone.

**4. Not a Clinical Tool**
This model is built for educational purposes only. It is **not validated for clinical use** and should not be used to make or inform real medical diagnoses. Any real-world application would require rigorous clinical validation, regulatory approval, and bias auditing across demographic subgroups.




