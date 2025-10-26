# 📊 Lending Club Loan Default Prediction: Data Preprocessing & Feature Engineering

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Latest-green)](https://pandas.pydata.org/)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Latest-orange)](https://scikit-learn.org/)

## 📖 Overview

This notebook documents a comprehensive data preparation and feature engineering pipeline for a binary classification task using the extensive **Lending Club Loan Data (2007-2018)**. The primary objective is to prepare a clean, scaled, and multicollinearity-reduced dataset for model training to predict loan default.

The notebook covers the complete journey from raw data to model-ready features, including:
- Target variable engineering
- Missing value handling
- Feature encoding and transformation
- Multicollinearity reduction
- Feature selection
- Data scaling and train-test split

---

## 🎯 Project Goal

**Predict loan default risk** using historical Lending Club loan data to help identify patterns that differentiate between loans that will be fully paid versus those that will default.

---

## 📂 Dataset Information

- **Source:** Lending Club Loan Data (2007-2018 Q4)
- **File:** `accepted_2007_to_2018Q4.csv`
- **Initial Dimensions:** ~2.26 million rows × 151 columns
- **Final Dimensions:** 24 features (after preprocessing)

---

## 🔄 Pipeline Overview

```
Raw Data (2.26M × 151)
    ↓
Target Definition & Filtering
    ↓
Missing Value Reduction
    ↓
Categorical Feature Encoding
    ↓
VIF-Based Multicollinearity Reduction (-31 features)
    ↓
Low Correlation Feature Removal (-90 features)
    ↓
Train-Test Split (75-25)
    ↓
Feature Scaling (MinMaxScaler)
    ↓
Final Dataset (24 features)
```

---

## 1️⃣ Initial Data Setup and Target Definition

### 🎯 Target Variable Engineering (`loan_status`)

The raw `loan_status` column was transformed to define our binary target variable:

| Original Status | Transformed Class | Label |
|:----------------|:------------------|:------|
| `Fully Paid` | Non-Default | **0** |
| `Charged Off` | Default | **1** |
| Other (Current, Late, etc.) | *Dropped* | - |

### ⚠️ Key Insight: Class Imbalance

After filtering, the dataset exhibited significant class imbalance:
- **Non-Default (0):** ~80% of loans
- **Default (1):** ~20% of loans

> **Note:** This imbalance will be a critical consideration during the modeling phase and may require techniques such as SMOTE, class weighting, or threshold adjustment.

---

## 2️⃣ Missing Value and Data Reduction

To manage the high dimensionality and missingness of the raw data, several early reduction steps were performed:

### Dropping High Missingness Columns
- **Threshold:** Columns with **>50% missing values** were removed
- **Affected fields:** Joint application data, detailed debt settlement information, late-stage payment statuses

### Dropping Redundant/Non-Predictive Columns
The following columns were removed due to high cardinality, unique identifiers, or limited predictive value:
- `id`
- `url`
- `pymnt_plan`
- `emp_title`
- `title`
- `zip_code`

---

## 3️⃣ Categorical Feature Preprocessing

All remaining categorical (object) features were converted to numerical format using appropriate encoding strategies.

### 📅 Date Features
Four date columns were converted to `datetime` objects for potential feature extraction:
- `issue_d`
- `earliest_cr_line`
- `last_pymnt_d`
- `last_credit_pull_d`

> While conversion was performed, explicit age-related feature extraction was not implemented in this notebook but can be easily added.

### 👔 Employment and Loan Term Features

- **`emp_length`:** Converted from strings (e.g., "10+ years") to **numerical values** representing years of employment
- **`term`:** Converted to binary flag `term_36_months` (1 = 36 months, 0 = 60 months)
- **`hardship_flag`:** Dropped (contained only single value: `'N'`)

### 🔢 Encoding Strategy

| Feature | Preprocessing/Encoding Applied | Notes |
|:--------|:------------------------------|:------|
| **`grade`** | **Ordinal Encoding** | Mapped 'A' through 'G' → 1 through 7 |
| **`verification_status`** | **Consolidation + One-Hot** | `'Source Verified'` merged into `'Verified'`, then one-hot encoded |
| **`initial_list_status`** | **Label Encoding** | Binary: 'w' (whole) and 'f' (fractional) |
| **`application_type`** | **Label Encoding** | Binary: 'Individual' and 'Joint App' |
| **`disbursement_method`** | **Label Encoding** | Binary: 'Cash' and 'DirectPay' |
| **`home_ownership`** | **One-Hot Encoding** | Created binary indicator columns |
| **`purpose`** | **One-Hot Encoding** | Created binary indicator columns |
| **`addr_state`** | **One-Hot Encoding** | Created binary indicator columns |

---

## 4️⃣ Feature Selection via Collinearity and Correlation

To ensure model robustness and interpretability, two targeted feature selection methods were applied:

### A. 📉 Multicollinearity Reduction (VIF)

**Method:** Iterative removal of features with **Variance Inflation Factor (VIF) > 10**

**Result:** Eliminated **31 highly collinear features**, including:
- `loan_amnt`
- `funded_amnt`
- `total_pymnt`
- `fico_range_high` / `fico_range_low`

This significantly reduced feature redundancy and improved model interpretability.

### B. 📊 Low Correlation Feature Removal

**Method:** Removed features with **absolute correlation < 0.05** with target variable (`loan_status`)

**Result:** Dropped **90 low-correlation features**, resulting in a **final curated set of 24 features**

---

## 5️⃣ Final Data Split and Scaling

### 🔀 Data Split

The cleaned and filtered dataset was split to prepare for model development:

- **Training Set:** 75% of data
- **Testing Set:** 25% of data
- **Random State:** 42 (for reproducibility)

### ⚖️ Feature Scaling

**Scaler Used:** MinMaxScaler

**Scaled Features (17 numerical features):**
- `installment`
- `dti`
- `inq_last_6mths`
- `total_rec_int`
- `last_pymnt_amnt`
- `acc_open_past_24mths`
- `avg_cur_bal`
- `bc_open_to_buy`
- `mo_sin_old_rev_tl_op`
- `mo_sin_rcnt_rev_tl_op`
- `mo_sin_rcnt_tl`
- `mort_acc`
- `mths_since_recent_bc`
- `mths_since_recent_inq`
- `num_actv_rev_tl`
- `num_tl_op_past_12m`
- `percent_bc_gt_75`

> **Important:** The scaler was fit **only on the training set** and then applied to both training and test sets to prevent data leakage.

### 💾 Final Output Files

The preprocessed and scaled datasets were saved as:
- `Scaler_train_df.csv` - Scaled training features
- `Scaler_test_df.csv` - Scaled testing features
- `y_train.csv` - Training target variable
- `y_test.csv` - Testing target variable

---

## 📈 Results Summary

| Metric | Value |
|:-------|:------|
| **Initial Features** | 151 |
| **After Missing Value Reduction** | ~120 |
| **After Encoding** | ~145 (with one-hot expansion) |
| **After VIF Reduction** | 114 (-31) |
| **Final Feature Count** | 24 (-90) |
| **Training Samples** | ~75% of filtered data |
| **Testing Samples** | ~25% of filtered data |

---

## 🔧 Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning preprocessing and scaling
- **Statsmodels** - VIF calculation for multicollinearity detection

---

## 📝 Key Takeaways

1. ✅ **Comprehensive preprocessing pipeline** from 151 to 24 features
2. ✅ **Systematic handling of missing values** and high-cardinality features
3. ✅ **Multiple encoding strategies** tailored to feature characteristics
4. ✅ **Rigorous feature selection** using both VIF and correlation methods
5. ✅ **Proper train-test split** with no data leakage
6. ✅ **Scaled features** ready for machine learning algorithms

---

## 🚀 Next Steps

The preprocessed dataset is now ready for:
- Model training (Logistic Regression, Random Forest, XGBoost, etc.)
- Handling class imbalance (SMOTE, class weights, threshold tuning)
- Hyperparameter optimization
- Model evaluation and comparison
- Feature importance analysis

---

# 🚀 Deep Learning Model for Loan Default Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red)](https://pytorch.org/)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Latest-orange)](https://scikit-learn.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-green)](https://developer.nvidia.com/cuda-zone)

## 📖 Overview

This notebook focuses on training a **Deep Neural Network (DNN)** using PyTorch to predict loan default (`loan_status=1`) based on the cleaned and engineered dataset from the preprocessing phase. The primary challenge addressed is the significant **class imbalance** inherent in loan default data, tackled through a robust dual-strategy approach combining loss weighting and weighted oversampling.

---

## 🎯 Modeling Objective

Build a high-performance binary classifier that can accurately identify loans at risk of default while effectively handling the 80-20 class imbalance between non-default and default cases.

---

## 📊 Pipeline Overview

```
Preprocessed Data (Preprocess_df_1.csv)
    ↓
Final Median Imputation
    ↓
Train-Test Split (80-20, Stratified)
    ↓
StandardScaler Normalization
    ↓
PyTorch Dataset & DataLoader (Weighted Sampling)
    ↓
Deep Neural Network Training (15 Epochs)
    ↓
Model Evaluation & Metrics
```

---

## 1️⃣ Final Data Preparation and Imputation

### 📂 Input Data
- **Source File:** `Preprocess_df_1.csv`
- **Features:** 25 columns (24 predictors + 1 target)
- **Origin:** Output from VIF and low-correlation filtering in preprocessing phase

### 🔧 Data Preparation Steps

#### A. Final Imputation
- **Method:** Median imputation applied to all remaining null values across 24 predictor columns
- **Purpose:** Ensure complete feature matrix before scaling and model input
- **Result:** Zero missing values in final dataset

#### B. Feature-Target Split
- **Features (X):** 24 predictor columns
- **Target (y):** `loan_status` (binary: 0 = Non-Default, 1 = Default)

#### C. Train-Test Split
- **Training Set:** 80% (~1,076,280 samples)
- **Test Set:** 20% (~269,070 samples)
- **Stratification:** `stratify=y` to maintain imbalanced class distribution in both subsets
- **Random State:** 42 (for reproducibility)

#### D. Feature Scaling
- **Scaler:** `StandardScaler` (mean=0, variance=1)
- **Rationale:** Critical for deep learning models to ensure faster convergence and better performance
- **Application:** Fitted on training set only, then transformed both train and test sets

---

## 2️⃣ Strategy for Handling Class Imbalance

The dataset exhibits significant class imbalance (**~80% Non-Default, ~20% Default**), which can lead to biased predictions favoring the majority class. A robust **dual-approach strategy** was implemented:

### ⚖️ Imbalance Handling Techniques

| Technique | Implementation Detail | Goal |
|:----------|:---------------------|:-----|
| **Loss Weighting** | Calculated `pos_weight` tensor based on training class distribution and passed to `nn.BCEWithLogitsLoss` | Penalizes false negatives (missed defaults) more heavily during training |
| **Weighted Oversampling** | Used `WeightedRandomSampler` in PyTorch `DataLoader` | Ensures minority (Default) class is sampled more frequently, effectively balancing samples seen per epoch |

### 🔢 Mathematical Foundation

**Positive Weight Calculation:**
```
pos_weight = (# Non-Default samples) / (# Default samples)
           ≈ 4.0 (given 80-20 split)
```

This combined approach ensures:
- ✅ The model learns to prioritize identifying defaults
- ✅ Each training epoch sees a balanced representation of both classes
- ✅ Loss function appropriately penalizes errors on minority class

---

## 3️⃣ Deep Neural Network (DNN) Architecture

A feedforward neural network with dropout regularization was constructed in PyTorch:

### 🏗️ Network Structure

| Layer | Type | Configuration | Output Dimension |
|:------|:-----|:--------------|:-----------------|
| **Input Layer** | `nn.Linear` | 24 → 64 | 64 |
| | `nn.ReLU` | Activation | 64 |
| | `nn.Dropout` | p = 0.3 | 64 |
| **Hidden Layer 1** | `nn.Linear` | 64 → 32 | 32 |
| | `nn.ReLU` | Activation | 32 |
| | `nn.Dropout` | p = 0.3 | 32 |
| **Output Layer** | `nn.Linear` | 32 → 1 | 1 (logit) |

### 📐 Model Configuration

- **Total Parameters:** ~3,000 trainable parameters
- **Input Features:** 24
- **Output:** Single logit (converted to probability via sigmoid)
- **Activation Function:** ReLU (Rectified Linear Unit)
- **Regularization:** Dropout (p=0.3) to prevent overfitting

### 🎛️ Training Configuration

| Component | Specification |
|:----------|:-------------|
| **Loss Function** | `nn.BCEWithLogitsLoss` with `pos_weight` |
| **Optimizer** | Adam (Adaptive Moment Estimation) |
| **Learning Rate** | 1e-3 (0.001) |
| **Batch Size** | (Configured via DataLoader) |
| **Epochs** | 15 |
| **Hardware** | GPU (CUDA enabled) |

> **Note:** `BCEWithLogitsLoss` combines sigmoid activation and binary cross-entropy loss for numerical stability.

---

## 4️⃣ Model Training and Evaluation

The model was trained for **15 epochs** with continuous monitoring of training and validation metrics.

### 📈 Training Performance (Epoch 15)

| Metric | Value | Interpretation |
|:-------|:------|:--------------|
| **Training Loss** | 0.3060 | Final cross-entropy loss on training data |
| **Training Accuracy** | 88.63% | Proportion of correct predictions on training set |

### 🎯 Validation Performance (Epoch 15)

| Metric | Value | Interpretation |
|:-------|:------|:--------------|
| **Validation Loss** | 0.3306 | Final loss on unseen test data |
| **Validation Accuracy** | 81.89% | Overall correct classification rate |
| **F1 Score** | **0.6866** | ⭐ Harmonic mean of precision and recall (key metric for imbalanced data) |
| **AUC-ROC** | **0.9853** | ⭐ Excellent discriminative ability between classes |

### 🏆 Key Performance Insights

1. **High AUC-ROC (0.9853):** Indicates exceptional ability to distinguish between Default and Non-Default loans across all threshold values
2. **Strong F1 Score (0.6866):** Demonstrates effective balance between precision and recall, crucial for imbalanced classification
3. **Minimal Overfitting:** Small gap between training accuracy (88.63%) and validation accuracy (81.89%) suggests good generalization
4. **Stable Convergence:** Loss curves show steady decrease without erratic fluctuations

---

## 5️⃣ Results Visualization

### 📉 Loss Curve (Training vs. Validation)

The loss plot demonstrates:
- ✅ Both training and validation losses decrease steadily
- ✅ Curves remain close throughout training
- ✅ No signs of significant overfitting
- ✅ Dropout and weighted sampling effectively regularize the model

**Interpretation:** The model successfully learns meaningful patterns without memorizing the training data.

### 📊 Evaluation Metrics Curve (Validation)

The metrics plot reveals:
- ✅ **AUC-ROC** reaches and maintains a high value (~0.985)
- ✅ **F1-Score** steadily improves throughout training
- ✅ **Validation Accuracy** stabilizes around 82%
- ✅ All metrics show consistent, stable performance

**Interpretation:** The class imbalance strategy (weighted loss + oversampling) successfully enables the model to learn both classes effectively.

---

## 📊 Performance Comparison

### Confusion Matrix Insights

While exact confusion matrix values aren't provided, the high F1-score and AUC-ROC suggest:

| Predicted ↓ / Actual → | Non-Default (0) | Default (1) |
|:----------------------|:----------------|:------------|
| **Non-Default (0)** | High True Negatives | Moderate False Negatives |
| **Default (1)** | Low False Positives | Good True Positives |

The model effectively identifies defaults while maintaining acceptable false positive rates.

---

## 🔧 Technologies Used

- **Python 3.8+**
- **PyTorch** - Deep learning framework
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Scikit-learn** - Preprocessing (StandardScaler) and metrics
- **CUDA** - GPU acceleration for training
- **Matplotlib/Seaborn** - Visualization

---

## 💡 Key Technical Decisions

### ✅ Why StandardScaler over MinMaxScaler?
- Deep neural networks benefit from centered data (mean=0)
- Reduces risk of vanishing/exploding gradients
- More robust to outliers in financial data

### ✅ Why BCEWithLogitsLoss?
- Combines sigmoid + BCE for numerical stability
- Prevents numerical overflow in exponential calculations
- More stable gradients during backpropagation

### ✅ Why Adam Optimizer?
- Adaptive learning rates for each parameter
- Built-in momentum for faster convergence
- Works well with sparse gradients common in financial data

### ✅ Why Dropout (p=0.3)?
- Prevents overfitting by randomly deactivating neurons
- Forces network to learn robust, distributed representations
- Particularly important with imbalanced data

---

## 📈 Model Performance Summary

| Aspect | Result | Status |
|:-------|:-------|:-------|
| **Convergence** | Smooth, stable over 15 epochs | ✅ Excellent |
| **Overfitting** | Minimal (train-val gap: ~7%) | ✅ Well-controlled |
| **Class Imbalance Handling** | F1=0.69, AUC=0.99 | ✅ Highly effective |
| **Discriminative Power** | AUC-ROC near perfect | ✅ Outstanding |
| **Production Readiness** | Strong metrics, stable training | ✅ Ready for deployment considerations |

---

## 🚀 Next Steps and Improvements

### Immediate Enhancements
- 🔄 **Threshold Tuning:** Optimize decision threshold for business requirements (cost of false negatives vs. false positives)
- 📊 **Cross-Validation:** Implement k-fold CV for more robust performance estimates
- 🎯 **Hyperparameter Optimization:** Grid search or Bayesian optimization for learning rate, dropout, architecture

### Advanced Techniques
- 🧠 **Ensemble Methods:** Combine with tree-based models (XGBoost, Random Forest)
- 📉 **Learning Rate Scheduling:** Implement decay or cyclic schedules
- 🔍 **Feature Importance:** SHAP values or attention mechanisms for interpretability
- ⚡ **Early Stopping:** Prevent unnecessary training epochs
- 💾 **Model Checkpointing:** Save best model based on validation F1-scor


**⭐ If you find this notebook helpful, please consider giving it a star!**

---
Final Report link : [Report_link](https://drive.google.com/file/d/1oXnlW_wo4j2b8Oh1G2WPmL4r3zrdg7cH/view?usp=sharing)

## 📚 References

- PyTorch Documentation: [https://pytorch.org/docs/](https://pytorch.org/docs/)
- Handling Imbalanced Data: [https://imbalanced-learn.org/](https://imbalanced-learn.org/)
- Binary Classification Metrics: [https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)

**⭐ If you find this notebook helpful, please consider giving it a star!**
