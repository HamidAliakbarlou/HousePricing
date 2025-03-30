# 🏡 House Prices – All-in-One Model Training and Comparison

This project explores multiple machine learning models to predict house prices using the Kaggle House Prices dataset. It covers everything from data preprocessing to feature engineering and model evaluation.

---

## 🔷 Essential Steps

| Step | Description |
|------|-------------|
| **1. Data Loading** | Load training and test datasets using `pandas.read_csv()` from the Kaggle housing dataset. |
| **2. Preprocessing** | Drop the `Id` column. Split data into numeric and categorical features. |
| **3. EDA** | Analyze correlation with the target (`SalePrice`). Visualize with heatmaps and scatter plots. |
| **4. Feature Engineering** | Create new features like `Bsmt` and `TotalPorchSF`. Remove low-correlation features. |
| **5. Null Handling** | Fill missing values with median or 0. |
| **6. Scaling** | Apply `StandardScaler` or `RobustScaler` to numerical features. |
| **7. Encoding** | Encode categorical variables using `LabelEncoder` and `get_dummies()`. |
| **8. Final Dataset** | Combine processed features for modeling. |

---

## 🤖 Models Implemented

| Model | Key Library | Notes |
|-------|-------------|-------|
| **Random Forest Regressor** | `sklearn.ensemble.RandomForestRegressor` | Good baseline for non-linear problems. |
| **XGBoost Regressor** | `xgboost.XGBRegressor` | High-performing boosting model. |
| **Ridge Regression** | `sklearn.linear_model.RidgeCV` | Regularized linear model (L2). |
| **Lasso Regression** | `sklearn.linear_model.LassoCV` | Regularized linear model (L1); also does feature selection. |

---

## 🔧 Fine-Tuning Tips

### 🔸 Random Forest
```python
RandomForestRegressor(n_estimators=100, random_state=42)
```
Tune `n_estimators`, `max_depth`, `min_samples_split`, etc.

### 🔸 XGBoost
```python
XGBRegressor(n_estimators=100, learning_rate=0.05)
```
Tune `max_depth`, `learning_rate`, `subsample`, `gamma`, etc.

### 🔸 Ridge & Lasso
```python
RidgeCV(alphas=np.logspace(-6, 6, 13), cv=10)
LassoCV(alphas=np.logspace(-6, 6, 13), cv=10)
```
Best for interpretable and sparse solutions.

---

## 📌 Highlights

| Feature | Description |
|---------|-------------|
| **Feature Selection** | Top 15 correlated variables selected. |
| **Visualization** | Heatmaps, scatter plots, and joint plots for data exploration. |
| **Evaluation** | R² and MAE used to assess model performance. |
| **Submission** | Predictions prepared for Kaggle competition submission. |

---

## ✅ Model Recommendation

| Scenario | Use |
|----------|-----|
| Best Accuracy | **XGBoost** |
| Simplicity | **Ridge** or **Random Forest** |
| Feature Reduction | **Lasso** |


