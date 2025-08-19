# Stroke Prediction Model: A Machine Learning Approach

This project focuses on building and evaluating various machine learning models to predict the likelihood of a patient experiencing a stroke based on their demographic and health data. The goal is to compare different classification algorithms, from simple models to advanced ensemble techniques, to find the most accurate and reliable predictor.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Data Preprocessing](#data-preprocessing)
- [Machine Learning Models](#machine-learning-models)
  - [Baseline Models](#baseline-models)
  - [Advanced Ensemble Techniques](#advanced-ensemble-techniques)
- [Evaluation](#evaluation)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [References](#references)

---

## Project Overview

Stroke is one of the leading causes of death and disability worldwide. Early prediction of stroke risk can help in timely intervention and better healthcare management. This project uses machine learning to analyze patient health and demographic data to predict the likelihood of stroke.

---

## Data Preprocessing

Before training the models, the following preprocessing steps were performed:

1. **Handling Missing Values**  
   The `bmi` column had missing values. These were filled using **K-Nearest Neighbors (KNN) Imputation**, which estimates missing values based on the 'k' most similar instances.

2. **Encoding Categorical Data**  
   Non-numeric features (e.g., `gender`, `work_type`) were converted into numerical format using **Label Encoding**.

3. **Handling Class Imbalance**  
   The dataset is highly imbalanced. **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to generate synthetic examples for the minority class (stroke cases).

4. **Feature Scaling**  
   For models sensitive to feature scale (e.g., Logistic Regression, SVM, KNN), **StandardScaler** was used to standardize the features.

---

## Machine Learning Models

### Baseline Models

1. **Logistic Regression**  
   - Finds a linear boundary to separate classes.  
   - Simple, fast, and highly interpretable.

2. **K-Nearest Neighbors (KNN)**  
   - Classifies based on the majority class of nearest neighbors.  
   - Good for datasets with irregular decision boundaries.

3. **Support Vector Machine (SVM)**  
   - Finds an optimal hyperplane with maximum margin.  
   - Effective in high-dimensional spaces.

4. **Naive Bayes**  
   - Probabilistic model based on Bayes’ theorem.  
   - Assumes feature independence, fast and efficient.

5. **Decision Tree**  
   - Creates a tree of "if-then-else" rules.  
   - Easy to interpret, captures non-linear relationships.

6. **Random Forest**  
   - Ensemble of Decision Trees with majority voting.  
   - Reduces overfitting, robust, and highly accurate.

### Advanced Ensemble Techniques

1. **Bagging (Bootstrap Aggregating)**  
   - Trains the same model on multiple random subsets of data.  
   - Reduces variance and improves generalization.  
   - Random Forest is a specific implementation of bagging.

2. **Boosting**  
   - Sequentially trains models to correct errors of previous models.  
   - Includes **AdaBoost** and **Gradient Boosting**.  
   - Reduces bias and increases accuracy.

3. **Stacking**  
   - Combines predictions from multiple base models using a meta-model.  
   - Leverages the strengths of different classifiers to improve performance.

---

## Evaluation

Models are evaluated based on standard classification metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Stacking ensemble was found to provide the best overall performance.

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/stroke-prediction.git
cd stroke-prediction
