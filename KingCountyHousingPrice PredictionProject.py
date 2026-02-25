# ==========================================================
#  King County Housing Price Prediction — Full Pipeline (Enhanced)
#  Author: Xin Wen
#  Purpose: Midterm Project - Data Science & Machine Learning
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# -----------------------------
# 1. Load and preprocess data
# -----------------------------
train = pd.read_csv('training_dataset-1.csv')
test = pd.read_csv('test_dataset-1.csv')

train.columns = [c.strip().lower() for c in train.columns]
test.columns = [c.strip().lower() for c in test.columns]

for df in [train, test]:
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df.drop(columns=['date'], inplace=True)

id_col = 'id'
target = 'price'
y = np.log1p(train[target])  # log-transform target
X = train.drop(columns=[target, id_col])

categorical_cols = ['zipcode']
numeric_cols = [c for c in X.columns if c not in categorical_cols]

# -----------------------------
# 2. Correlation Analysis
# -----------------------------
train_corr = train.copy()
train_corr['log_price'] = np.log1p(train_corr['price'])
corr = train_corr[numeric_cols + ['log_price']].corr()['log_price'].drop('log_price').sort_values(key=lambda s: s.abs(), ascending=False)

plt.figure(figsize=(7,5))
sns.barplot(x=corr.values[:10], y=corr.index[:10], palette='viridis')
plt.title('Top 10 Correlated Features with log(price)')
plt.xlabel('Pearson Correlation')
plt.tight_layout()
plt.savefig('top10_corr_logprice.png', dpi=300)
plt.close()

# -----------------------------
# 3. Multicollinearity Check
# -----------------------------
heat = train[numeric_cols].corr()
plt.figure(figsize=(9,7))
sns.heatmap(heat, cmap='coolwarm', center=0)
plt.title('Numeric Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('corr_heatmap_numeric.png', dpi=300)
plt.close()

# -----------------------------
# 4. Preprocessing
# -----------------------------
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# -----------------------------
# 5. Model Selection & Comparison
# -----------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

cv_results = {}
for name, model in models.items():
    pipe = Pipeline([('preprocess', preprocess), ('model', model)])
    scores = cross_val_score(pipe, X, y, scoring='neg_root_mean_squared_error', cv=5)
    cv_results[name] = -scores.mean()
    print(f"{name}: RMSE(log price) = {-scores.mean():.4f}")

# Create comparison DataFrame
cv_df = pd.DataFrame.from_dict(cv_results, orient='index', columns=['CV_RMSE_log'])
cv_df.sort_values('CV_RMSE_log', inplace=True)
cv_df.to_csv('model_comparison.csv')

# Plot model comparison
plt.figure(figsize=(7,4))
sns.barplot(x=cv_df['CV_RMSE_log'], y=cv_df.index, palette='Blues_d')
plt.title('Model Comparison (Lower RMSE = Better)')
plt.xlabel('Cross-Validation RMSE (log)')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
plt.close()

# -----------------------------
# 6. Train Best Model (GBR)
# -----------------------------
best_model = GradientBoostingRegressor(random_state=42)
best_pipe = Pipeline([('preprocess', preprocess), ('model', best_model)])
best_pipe.fit(X, y)

# -----------------------------
# 7. Feature Importance
# -----------------------------
oh = best_pipe.named_steps['preprocess'].named_transformers_['cat']
cat_features = list(oh.get_feature_names_out(categorical_cols))
feature_names = numeric_cols + cat_features
importances = best_pipe.named_steps['model'].feature_importances_

feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)
feat_imp.to_csv('feature_importances.csv', index=False)

plt.figure(figsize=(8,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp.head(15), palette='plasma')
plt.title('Top 15 Feature Importances (Gradient Boosting)')
plt.tight_layout()
plt.savefig('top15_feature_importances.png', dpi=300)
plt.close()

# -----------------------------
# 8. Predict on Test Set
# -----------------------------
preds = best_pipe.predict(test.drop(columns=[id_col]))
preds = np.expm1(preds)  # reverse log
preds = np.maximum(preds, 0)

submission = pd.DataFrame({'id': test[id_col], 'price': preds})
submission.to_csv('predictions_submission.csv', index=False, encoding='utf-8')

# -----------------------------
# 9. Summary Output
# -----------------------------
summary = pd.DataFrame({
    'Metric': ['Best Model', 'CV RMSE (log)', 'Top Features'],
    'Value': ['Gradient Boosting Regressor',
              f"{cv_df.loc['Gradient Boosting','CV_RMSE_log']:.3f}",
              ', '.join(feat_imp.head(10)['Feature'].tolist())]
})
summary.to_csv('model_summary.csv', index=False)

print("\n=== Model Comparison ===")
print(cv_df)
print("\n=== Best Model Summary ===")
print(summary)
print("\nSubmission file created: predictions_submission.csv")
