import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

cols_fillna_none = [
    'Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
    'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 
    'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature'
]

for col in cols_fillna_none:
    train[col] = train[col].fillna('None')
    test[col] = test[col].fillna('None')


train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
test['LotFrontage'] = test.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.median())


train_id = train['Id']
test_id = test['Id']
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)


y = train['SalePrice']
X = train.drop('SalePrice', axis=1)

print("data dimensions after preprocessing:", X.shape)

train_df = pd.read_csv('train.csv')

numeric_cols = train_df.select_dtypes(include=[np.number]).columns
train_df[numeric_cols] = train_df[numeric_cols].fillna(train_df[numeric_cols].median())
full_corr_matrix = train_df[numeric_cols].corr()

top_k = 10
top_features = full_corr_matrix.nlargest(top_k, 'SalePrice')['SalePrice'].index
correlation_subset = train_df[top_features].corr()

plt.figure(figsize=(10, 8))
sns.set(font_scale=1.1)
heatmap = sns.heatmap(correlation_subset, 
                      annot=True, 
                      fmt='.2f', 
                      cmap='YlGnBu', 
                      square=True, 
                      linewidths=0.5)

plt.title('Correlation Heat Map: Top 10 Features')
plt.savefig('correlation_heatmap.png')

print(full_corr_matrix['SalePrice'].sort_values(ascending=False).head(top_k))


train = pd.read_csv('train.csv')

top_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 
                'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']


X = train[top_features].fillna(train[top_features].median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
var_ratio = pca.explained_variance_ratio_

fa = FactorAnalysis(n_components=3, random_state=42)
X_fa = fa.fit_transform(X_scaled)


plt.figure(figsize=(10, 5))
plt.bar(range(1, 4), var_ratio, alpha=0.6, label='Individual Variance', color='g')
plt.step(range(1, 4), np.cumsum(var_ratio), where='mid', label='Cumulative Variance')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.title('Step 3: PCA Variance Analysis')
plt.legend(loc='best')
plt.grid(axis='y', linestyle='--')
plt.savefig('pca_variance.png')
plt.show() 

pca_loadings = pd.DataFrame(pca.components_.T, index=top_features, columns=['PC1', 'PC2', 'PC3'])
plt.figure(figsize=(10, 6))
sns.heatmap(pca_loadings, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Loadings on Principal Components')
plt.savefig('pca_loadings.png') 

print(f"Total variance explained by 3 components: {np.sum(var_ratio):.4f}")

fa_loadings = pd.DataFrame(fa.components_.T, index=top_features, columns=['Factor 1', 'Factor 2', 'Factor 3'])

plt.figure(figsize=(10, 8))
sns.heatmap(fa_loadings, annot=True, cmap='RdBu_r', center=0)
plt.title('Factor Analysis: Feature Loadings Heatmap')
plt.show()

plt.figure(figsize=(10, 7))
plt.scatter(X_fa[:, 0], X_fa[:, 1], c=train['SalePrice'], cmap='viridis')
plt.colorbar(label='SalePrice')
plt.xlabel('Factor 1 (Latent Quality/Size)')
plt.ylabel('Factor 2 (Latent Interior Space)')
plt.title('House Distribution in Factor Space')
plt.show()



X_encoded = pd.get_dummies(X) 
test_encoded = pd.get_dummies(test)

X_encoded, test_encoded = X_encoded.align(test_encoded, join='left', axis=1, fill_value=0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
test_scaled = scaler.transform(test_encoded)

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, max_depth=5)

k_values = [3, 5, 10]
results_cv = {}

print("--- Evaluation with K-Fold ---")
for k in k_values:
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_scores = -cross_val_score(model, X_scaled, y, scoring='neg_mean_squared_error', cv=kf)
    rmse_scores = np.sqrt(mse_scores)
    results_cv[k] = rmse_scores.mean()
    print(f"K={k}: Average RMSE = {rmse_scores.mean():.2f}")

model.fit(X_scaled, y)
predictions_train = model.predict(X_scaled)

mse = mean_squared_error(y, predictions_train)
mae = mean_absolute_error(y, predictions_train)
rmse = np.sqrt(mse)

print("\n--- Final Metrics on Train Set ---")
print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

model.fit(X_pca, y) 
pca_preds = model.predict(X_pca)
pca_rmse = np.sqrt(mean_squared_error(y, pca_preds))
print(f"\nRMSE with PCA (3 components): {pca_rmse:.2f}")
print(f"Analysis: Using only 3 components significantly {'increased' if pca_rmse > rmse else 'decreased'} the error.")


final_test_preds = model.fit(X_scaled, y).predict(test_scaled)
test_original = pd.read_csv('test.csv') 
test_original['predicted price'] = final_test_preds
test_original.to_csv('test_with_predictions.csv', index=False)