# House Price Prediction — XGBoost + PCA / Factor Analysis

Regression on the Ames Housing dataset to predict `SalePrice`, with dimensionality-reduction analysis.

## Pipeline

### 1. Preprocessing
- Fills "no feature" categorical columns (`Alley`, `Bsmt*`, `Garage*`, `PoolQC`, …) with `'None'`.
- Imputes `LotFrontage` with the median per `Neighborhood`.
- Drops `Id`, separating `SalePrice` as the target `y`.

### 2. Correlation analysis
- Median-imputes numeric columns and computes the correlation matrix.
- Plots and saves a heatmap of the top 10 features correlated with `SalePrice` → `correlation_heatmap.png`.

### 3. PCA & Factor Analysis
On the top 9 engineered features (`OverallQual`, `GrLivArea`, `GarageCars`, `GarageArea`, `TotalBsmtSF`, `1stFlrSF`, `FullBath`, `TotRmsAbvGrd`, `YearBuilt`), after `StandardScaler`:
- **PCA** with 3 components — individual + cumulative explained variance plot (`pca_variance.png`) and a feature-loadings heatmap (`pca_loadings.png`).
- **Factor Analysis** with 3 components — loadings heatmap and a scatter of houses in latent factor space colored by `SalePrice`.

### 4. Modeling
- One-hot encodes features and aligns train/test columns.
- **XGBoost** regressor (`n_estimators=1000`, `learning_rate=0.05`, `max_depth=5`).
- Evaluated with **K-Fold** cross-validation (K = 3, 5, 10) using RMSE.
- Reports MSE / MAE / RMSE on the training set and compares RMSE using the 3 PCA components.
- Writes final test predictions to `test_with_predictions.csv`.

## Requirements

```
pandas numpy xgboost scikit-learn seaborn matplotlib
```

`train.csv` and `test.csv` (Ames Housing) are expected in the working directory.

## Run

```bash
python house-price-prediction.py
```

Outputs: `correlation_heatmap.png`, `pca_variance.png`, `pca_loadings.png`, `test_with_predictions.csv`.
