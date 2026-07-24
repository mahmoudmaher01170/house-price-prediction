# house-price-prediction
Real Estate Price Prediction using Machine Learning and XGBoost Regressor (R² = 0.9315)
# 🏡 Real Estate Price Prediction using Machine Learning

A machine learning project that predicts house prices based on various real estate features. The workflow covers full data preprocessing, feature engineering, missing value handling, one-hot encoding, and model training using **XGBoost Regressor**, achieving an **R² score of 0.9315**.

---

## 📸 Overview & Performance

* **Model Used:** XGBoost Regressor
* **Mean Squared Error (MSE):** `5,689,124,117.86`
* **Root Mean Squared Error (RMSE):** `75,426.28`
* **Mean Absolute Error (MAE):** `35,885.50`
* **R² Score:** `0.9315` (Explains ~93.15% of the variance in prices)

---

## 🛠️ Project Pipeline

1. **Data Cleaning & Preprocessing:**
   - Handled missing values using median/mode imputation.
   - Cleaned numerical features (`price`, `carpet_area_sqft`, `floor_num`).
   - Standardized textual and categorical entries.

2. **Feature Engineering & Encoding:**
   - Created scaled area features.
   - Applied One-Hot Encoding (`pd.get_dummies`) to handle categorical columns (`locality`, `facing`, etc.).

3. **Model Training & Evaluation:**
   - Split dataset into training and testing sets.
   - Trained an optimized **XGBoost Regressor**.
   - Evaluated performance using MSE, RMSE, MAE, and R² Score.

---

## ⚙️ Tech Stack & Libraries

- **Language:** Python 3.x
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`, `xgboost`
- **Visualization:** `matplotlib`, `seaborn`

---

## 📊 Results Summary

The XGBoost model demonstrates high precision in predicting real estate prices with minimal residual error across various price ranges.
