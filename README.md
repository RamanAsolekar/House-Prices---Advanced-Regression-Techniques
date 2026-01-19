# House Prices: Advanced Regression Techniques

This repository contains a machine learning solution for the **Kaggle House Prices - Advanced Regression Techniques** competition. The goal is to predict the final sales price of residential homes in Ames, Iowa, using 79 explanatory variables describing (almost) every aspect of the homes.

The solution implements a **Weighted Ensemble** approach, combining linear regularized models (Lasso, Ridge) with Gradient Boosting (XGBoost) to achieve robust predictions.

---

## 📊 Approach & Methodology

The solution follows a rigorous pipeline designed to handle the skewness of real-world data and maximize model generalization.

### 1. Data Preprocessing
- **Outlier Removal:** Removed extreme outliers in `GrLivArea` (> 4000 sq ft) with low prices to prevent skewing the regression.
- **Target Transformation:** Applied `log1p` (Log(1+x)) to the target variable (`SalePrice`) to normalize the distribution.
- **Missing Value Imputation:**
  - **Categorical (Pool, Garage, Basement):** Filled with "None" where missing implies the facility doesn't exist.
  - **Numerical (Area, Cars, Year):** Filled with `0` for missing amenities.
  - **LotFrontage:** Imputed using the median value of the specific **Neighborhood**.
  - **Mode Imputation:** Used for categorical features like `MSZoning`, `Electrical`, and `SaleType`.

### 2. Feature Engineering
- **Type Conversion:** Converted numerical categories (e.g., `MSSubClass`, `YrSold`, `MoSold`) into strings to treat them as categorical data.
- **Label Encoding:** Applied to ordinal features (e.g., `FireplaceQu`, `BsmtQual`, `KitchenQual`) to preserve rank information.
- **New Feature Creation:** Engineered `TotalSF` (Total Square Footage) by summing Basement, 1st Floor, and 2nd Floor areas.
- **Skewness Correction:** Applied **Box-Cox transformation** (lambda=0.15) to all numerical features with high skew (> 0.75) to improve linear model performance.
- **One-Hot Encoding:** Used `pd.get_dummies` to handle remaining categorical variables.

### 3. Modeling Strategy
A multi-model ensemble was used to capture both linear and non-linear patterns:

1.  **Lasso Regression (L1):** Uses `RobustScaler` to handle outliers. Good for feature selection.
2.  **Ridge Regression (L2):** Uses `RobustScaler`. Good for handling multicollinearity.
3.  **XGBoost Regressor:** A tree-based gradient boosting model tuned for depth (`max_depth=3`) and conservative learning (`learning_rate=0.05`).

### 4. Final Ensemble
The final prediction is a weighted average of the three models, optimized for the lowest Root Mean Squared Error (RMSE):

$$\text{Final Prediction} = 0.35(\text{Lasso}) + 0.35(\text{Ridge}) + 0.30(\text{XGBoost})$$

*Note: The final predictions are converted back to real prices using `np.expm1` (inverse log).*

---

## 🛠 Dependencies

The project requires Python 3.x and the following libraries:
- `pandas` & `numpy`: Data manipulation.
- `scikit-learn`: Preprocessing (LabelEncoder, RobustScaler) and Linear Models (Lasso, Ridge).
- `scipy`: Statistical transformations (Skew, BoxCox).
- `xgboost`: Gradient Boosting implementation.

---

## 🚀 How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Data:**
    Download `train.csv` and `test.csv` from the [Kaggle Competition Page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) and place them in the project root.

3.  **Run the Script:**
    Execute the notebook or script. The model will:
    - Train Lasso, Ridge, and XGBoost.
    - Output status messages during training.
    - Generate a file named `submission.csv`.

4.  **Submit:**
    Upload `submission_top_tier.csv` to Kaggle.