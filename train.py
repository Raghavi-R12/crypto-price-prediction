import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    df = pd.read_csv('crypto_prices.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("No dataset found. Generating placeholder data...")
    np.random.seed(42)
    data = {
        'Open_Price': np.random.uniform(100, 50000, 1079),
        'High_Price': np.random.uniform(105, 51000, 1079),
        'Low_Price': np.random.uniform(95, 49000, 1079),
        'Volume': np.random.uniform(10000, 10000000, 1079),
        'Target_Price': np.random.uniform(100, 50000, 1079)
    }
    df = pd.DataFrame(data)

X = df[['Open_Price', 'High_Price', 'Low_Price', 'Volume']]
y = df['Target_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=0.1),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Support Vector Regression (SVR)": SVR(kernel='rbf')
}

best_model = None
best_r2 = -float('inf')

print("\n--- Model Evaluation Results ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"\n[{name}]")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  Mean Absolute Error (MAE):     {mae:.4f}")
    print(f"  R2 Score (Explained Variance):  {r2:.4f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = model

if best_model is not None:
    joblib.dump(best_model, 'crypto_model.pkl')
    print(f"\n🎉 Best model saved to 'crypto_model.pkl' with R2 Score of {best_r2:.4f}")
