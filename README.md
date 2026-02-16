# 🌍 WHO-LIFE: Global Lifestyle Segmentation & Prediction
End-to-end ML pipeline for WHO-LIFE: Unsupervised K-Means clustering to identify global lifestyle profiles &amp; supervised Gradient Boosting to predict segments. Features EDA, PCA, KNN Imputation, &amp; GridSearch tuning. Validated via 10-fold Stratified CV for robust evaluation of health, financial &amp; behavioral patterns.

## 📌 Project Overview
Developed for the **WHO-LIFE Initiative**, this project analyzes individual-level data from global citizens to identify distinct lifestyle patterns and build a predictive model for targeted public health interventions.

By integrating health, financial, and behavioral data, we successfully segmented the population into actionable clusters and built a high-performance classifier to predict these segments for new individuals.


---

## 📁 Repository Structure
* **`Descriptive Modeling.ipynb`**: Unsupervised learning pipeline. Includes EDA, data cleaning, and **K-Means Clustering** to define lifestyle profiles.
* **`Predictive Modeling.ipynb`**: Supervised learning pipeline. Implements **Gradient Boosting**, Random Forest, and SVM with hyperparameter tuning.
* **`ML_Group07_Report.pdf`**: Final technical report detailing methodology, cluster definitions, and policy recommendations.
* **Dataset**: `world_citizens.csv` (15,977 records; 13 features including BMI, Spending Score, and Work-Life Balance).

---

## 🧪 Methodology

### 1. Data Engineering & Preprocessing
* **Missing Data:** Applied **KNN Imputation** to fill gaps while preserving local data structures.
* **Feature Enrichment:** Integrated external data like **GDP per capita** and **Population** data to contextualize behavioral patterns.
* **Scaling:** Used **RobustScaler** to handle outliers in skewed financial and health distributions.


### 2. Descriptive Modeling (Clustering)
Instead of dimensionality reduction, we created domain-specific **Feature Profiles** to run K-Means Clustering:
* **Health Profile:** BMI, Sleep Hours, Workout Frequency.
* **Financial Profile:** Income, Spending Score, Work-Life Balance.
* **Lifestyle Profile (Selected):** A hybrid approach that produced the most distinct and actionable segments (e.g., "Wellness-Oriented", "Digitally Engaged").


### 3. Predictive Modeling (Classification)
We built a robust classification engine to predict the identified lifestyle clusters:
* **Feature Selection:** Used **SelectKBest** (ANOVA F-value) to identify the most discriminative features.
* **Models Tested:** Logistic Regression, KNN, Random Forest, SVM, and **Gradient Boosting**.
* **Validation:** Implemented **10-Fold Stratified Cross-Validation** to ensure reliability.
* **Optimization:** Fine-tuned hyperparameters using `GridSearchCV`.

---

## 📊 Key Findings
* **Best Model:** **Gradient Boosting Classifier** achieved the highest accuracy and stability.
* **Cluster Insights:**
    * **Wellness-Oriented:** High fitness engagement, balanced sleep, moderate screen time.
    * **Digitally Engaged:** High screen usage, impulsive spending patterns, lower physical activity.
    * **Fitness-Driven:** Prioritizes workout frequency and nutritional consistency.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** `Pandas`, `NumPy`, `Scikit-learn`, `Matplotlib`, `Seaborn`.
---

## 👥 Team
* **António Santos**
* **Ashool Lakhani**
* **Francisco Oliveira**
* **Tara Kouros**
