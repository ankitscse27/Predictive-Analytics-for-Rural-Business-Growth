# --------------------------------------------------------------------------
# ADVANCED PREDICTIVE ANALYTICS FOR RURAL BUSINESS GROWTH
# --------------------------------------------------------------------------
# This script builds an XGBoost model to forecast sales for a small business,
# incorporating advanced feature engineering and hyperparameter tuning.
# --------------------------------------------------------------------------

# ## 1. IMPORT LIBRARIES ##
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ## 2. DATA GENERATION & ADVANCED FEATURE ENGINEERING ##

# --- Create a Sample Dataset ---
# In a real-world scenario, you would load your own sales data.
# For example: df = pd.read_csv('your_sales_data.csv')
print("Step 1: Generating sample data and engineering features...")
data = {
    'Date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=365, freq='D')),
    'Temperature': np.random.randint(10, 35, size=365),
    'Local_Event': np.random.choice([0, 1], size=365, p=[0.8, 0.2]), # 1 if there's an event
    'Is_Holiday': np.random.choice([0, 1], size=365, p=[0.9, 0.1]) # 1 if it's a holiday
}
df = pd.DataFrame(data)

# --- Advanced Feature Engineering ---

# 1. Time-Based Features
df['Month'] = df['Date'].dt.month
df['Day_of_Week'] = df['Date'].dt.dayofweek # Monday=0, Sunday=6
df['Week_of_Year'] = df['Date'].dt.isocalendar().week.astype(int)

# 2. Cyclical Features (to help the model understand the cyclical nature of time)
df['Month_sin'] = np.sin(2 * np.pi * df['Month']/12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month']/12)
df['Day_of_Week_sin'] = np.sin(2 * np.pi * df['Day_of_Week']/7)
df['Day_of_Week_cos'] = np.cos(2 * np.pi * df['Day_of_Week']/7)

# 3. Create Target Variable (Sales) and Lag Feature
# We generate synthetic sales data for this example.
base_sales = 100
month_effect = np.sin((df['Month'] - 3) * np.pi / 6) * 50
day_effect = (df['Day_of_Week'] >= 5) * 60 # Weekend boost
event_holiday_effect = (df['Local_Event'] + df['Is_Holiday']) * 80
noise = np.random.normal(0, 20, size=365)
df['Sales'] = (base_sales + month_effect + day_effect + event_holiday_effect + df['Temperature'] * 1.5 + noise).astype(int).clip(lower=20)

# 4. Lag Feature (a powerful predictor using yesterday's sales)
df['Sales_Lag_1'] = df['Sales'].shift(1)

# Drop the first row which has a NaN value from the lag feature
df = df.dropna()

print("Data Head with Advanced Features:")
print(df.head())
print("-" * 50)


# ## 3. MODEL TRAINING & HYPERPARAMETER TUNING ##

# --- Define Features (X) and Target (y) ---
features = [
    'Temperature', 'Local_Event', 'Is_Holiday', 'Week_of_Year',
    'Month_sin', 'Month_cos', 'Day_of_Week_sin', 'Day_of_Week_cos', 'Sales_Lag_1'
]
X = df[features]
y = df['Sales']

# --- Split Data into Training and Testing Sets ---
# We set shuffle=False because the order of data matters in time-series analysis.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# --- Hyperparameter Tuning with GridSearchCV ---
print("\nStep 2: Starting hyperparameter tuning for XGBoost model...")
# Define the model and the grid of parameters to search
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42)
param_grid = {
    'n_estimators': [100, 200],      # Number of trees in the forest
    'max_depth': [3, 5, 7],         # Maximum depth of a tree
    'learning_rate': [0.05, 0.1],   # Step size shrinkage
    'subsample': [0.8, 1.0]         # Fraction of samples to be used for fitting the individual base learners
}

# Set up the search with 3-fold cross-validation
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model from the search
best_model = grid_search.best_estimator_
print(f"\nBest Parameters Found: {grid_search.best_params_}")
print("-" * 50)


# ## 4. ADVANCED EVALUATION & INTERPRETATION ##

print("\nStep 3: Evaluating the model and interpreting results...")
# --- Evaluate the Best Model on the Test Set ---
y_pred = best_model.predict(X_test)

# Calculate key performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# --- Feature Importance ---
# This chart shows which factors have the biggest impact on sales predictions.
plt.figure(figsize=(12, 7))
feature_importances = pd.Series(best_model.feature_importances_, index=features)
feature_importances.nlargest(10).plot(kind='barh', color='skyblue')
plt.title('Top 10 Most Important Features for Sales Prediction')
plt.xlabel('Importance')
plt.gca().invert_yaxis() # To display the most important feature at the top
plt.tight_layout()
plt.show()
print("Feature importance plot has been generated.")
print("-" * 50)


# ## 5. MAKING PREDICTIONS & STRATEGIC DECISIONS ##

print("\nStep 4: Generating a sales forecast for the next 7 days...")
# --- Create Future Data for Prediction ---
# We need the last known sales value to create the lag feature for the first prediction day.
last_known_sales = df['Sales'].iloc[-1]

# Create a dataframe for the next 7 days
future_data = pd.DataFrame({
    'Temperature': [15, 16, 14, 18, 20, 22, 21],
    'Local_Event': [0, 0, 0, 0, 1, 1, 0], # A weekend event is planned
    'Is_Holiday':  [0, 0, 0, 0, 0, 0, 0],
    'Week_of_Year': [1, 1, 1, 1, 1, 1, 1],
    'Month_sin': [np.sin(2 * np.pi * 1/12)] * 7, # It's January
    'Month_cos': [np.cos(2 * np.pi * 1/12)] * 7,
    'Day_of_Week_sin': np.sin(2 * np.pi * np.arange(7)/7), # Monday to Sunday
    'Day_of_Week_cos': np.cos(2 * np.pi * np.arange(7)/7),
    'Sales_Lag_1': [last_known_sales, 0, 0, 0, 0, 0, 0] # Initialize with the last known value, others are placeholders
})

# --- Predict Sequentially ---
# We predict one day at a time, using the prediction of the previous day as the new lag feature.
predicted_sales = []
for i in range(len(future_data)):
    # Predict for the current day
    current_prediction = best_model.predict(future_data.iloc[[i]][features])[0]
    predicted_sales.append(int(current_prediction))
    
    # Update the lag feature for the *next* day in the dataframe, if it exists
    if i + 1 < len(future_data):
        future_data.loc[i + 1, 'Sales_Lag_1'] = current_prediction

future_data['Predicted_Sales'] = predicted_sales

print("\n--- Advanced Sales Forecast for Next Week ---")
print(future_data[['Predicted_Sales', 'Local_Event', 'Temperature', 'Day_of_Week_sin']])
print("\nScript finished.")