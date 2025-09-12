# --------------------------------------------------------------------------
# ENHANCED PREDICTIVE ANALYTICS FOR RURAL BUSINESS GROWTH
# --------------------------------------------------------------------------
# This script builds an advanced XGBoost model to forecast sales,
# incorporating sophisticated feature engineering, early stopping,
# and enhanced visualizations for deeper business insights.
# --------------------------------------------------------------------------

# ## 1. IMPORT LIBRARIES ##
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
# Use a consistent style for plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (15, 7)


def create_features(df):
    """
    Create time-series features from a datetime index.
    """
    df = df.copy()
    # 1. Standard Time-Based Features
    df['Month'] = df['Date'].dt.month
    df['Day_of_Week'] = df['Date'].dt.dayofweek  # Monday=0, Sunday=6
    df['Week_of_Year'] = df['Date'].dt.isocalendar().week.astype(int)

    # 2. Cyclical Features for better seasonality representation
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Day_of_Week_sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
    df['Day_of_Week_cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)

    return df

def generate_sample_data():
    """
    Generates a sample sales dataset with realistic patterns.
    """
    print("Step 1: Generating sample data...")
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=365 * 2, freq='D'))
    data = {
        'Date': dates,
        'Temperature': np.random.randint(10, 35, size=len(dates)),
        'Local_Event': np.random.choice([0, 1], size=len(dates), p=[0.85, 0.15]),
        'Is_Holiday': np.random.choice([0, 1], size=len(dates), p=[0.95, 0.05])
    }
    df = pd.DataFrame(data)

    # --- Create Target Variable (Sales) ---
    df_features = create_features(df.copy())
    base_sales = 150
    # More complex seasonal effect (summer peak, smaller holiday peak)
    month_effect = np.sin((df_features['Month'] - 6) * np.pi / 6) * -70 + 70
    day_effect = (df_features['Day_of_Week'] >= 5) * 80  # Stronger weekend boost
    event_holiday_effect = (df_features['Local_Event'] * 100 + df_features['Is_Holiday'] * 120)
    noise = np.random.normal(0, 25, size=len(dates))
    df['Sales'] = (base_sales + month_effect + day_effect + event_holiday_effect + df_features['Temperature'] * 2.0 + noise).astype(int).clip(lower=30)
    
    return df

def add_lag_and_rolling_features(df):
    """
    Adds lag and rolling window features which are powerful predictors.
    """
    print("Step 2: Engineering advanced lag and rolling features...")
    df = df.copy()
    df['Sales_Lag_1'] = df['Sales'].shift(1) # Previous day's sales
    df['Sales_Lag_7'] = df['Sales'].shift(7) # Sales from the same day last week
    df['Sales_Rolling_Mean_7'] = df['Sales'].shift(1).rolling(window=7).mean() # Avg sales of the last 7 days

    # Drop rows with NaN values created by shifts/rolling windows
    df = df.dropna()
    return df

def main():
    """
    Main function to run the entire forecasting pipeline.
    """
    # ## 2. DATA GENERATION & FEATURE ENGINEERING ##
    df = generate_sample_data()
    df = add_lag_and_rolling_features(df)
    df_features_final = create_features(df)

    print("\nData Head with All Features:")
    print(df_features_final.head())
    print("-" * 60)

    # ## 3. MODEL TRAINING & HYPERPARAMETER TUNING ##
    features = [
        'Temperature', 'Local_Event', 'Is_Holiday', 'Week_of_Year',
        'Month_sin', 'Month_cos', 'Day_of_Week_sin', 'Day_of_Week_cos',
        'Sales_Lag_1', 'Sales_Lag_7', 'Sales_Rolling_Mean_7'
    ]
    target = 'Sales'

    X = df_features_final[features]
    y = df_features_final[target]

    # Use a time-based split, no shuffling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    print("\nStep 3: Starting hyperparameter tuning for XGBoost model...")
    xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42)

    # A more focused parameter grid
    param_grid = {
        'n_estimators': [200, 500],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }

    grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"\nBest Parameters Found: {grid_search.best_params_}")
    print("-" * 60)

    # ## 4. ADVANCED EVALUATION & INTERPRETATION ##
    print("\nStep 4: Evaluating the final model and interpreting results...")
    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\n--- Model Evaluation ---")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"R-squared (R²): {r2:.2f} (Model explains {r2:.0%} of sales variance)")

    # --- Visualization 1: Feature Importance ---
    plt.figure()
    feature_importances = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=False)
    sns.barplot(x=feature_importances, y=feature_importances.index, palette='viridis')
    plt.title('Feature Importance for Sales Prediction')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

    # --- Visualization 2: Actual vs. Prediction on Test Set ---
    results = pd.DataFrame({'Actual Sales': y_test, 'Predicted Sales': y_pred})
    results.plot(title='Model Performance: Actual vs. Predicted Sales on Test Data', style=['-', '--'])
    plt.ylabel('Sales ($)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("-" * 60)
    
    # ## 5. MAKING PREDICTIONS FOR THE FUTURE ##
    print("\nStep 5: Generating a sales forecast for the next 14 days...")
    
    # Get the last 7 days of data to calculate future lag/rolling features
    last_window = df_features_final.iloc[-7:]
    
    future_predictions = []
    current_batch = last_window.copy()

    for i in range(14): # Predict for 14 days
        # Prepare the next day's features
        next_date = current_batch['Date'].iloc[-1] + pd.Timedelta(days=1)
        # Mock future exogenous variables
        next_temp = np.random.randint(15, 25)
        next_event = np.random.choice([0, 1], p=[0.9, 0.1])
        next_holiday = 0 # Assuming no holidays in the next 14 days

        # Create the feature row for prediction
        new_row = pd.DataFrame({
            'Date': [next_date], 'Temperature': [next_temp],
            'Local_Event': [next_event], 'Is_Holiday': [next_holiday]
        })
        
        # Add time-based features
        new_row = create_features(new_row)
        
        # Add lag/rolling features based on `current_batch`
        new_row['Sales_Lag_1'] = current_batch['Sales'].iloc[-1]
        new_row['Sales_Lag_7'] = current_batch['Sales'].iloc[0]
        new_row['Sales_Rolling_Mean_7'] = current_batch['Sales'].mean()
        
        # Predict
        prediction = best_model.predict(new_row[features])[0]
        future_predictions.append(prediction)
        
        # Update current_batch for the next iteration
        new_row['Sales'] = prediction
        current_batch = pd.concat([current_batch.iloc[1:], new_row], ignore_index=True)

    # --- Visualization 3: Final Forecast Plot ---
    forecast_dates = pd.date_range(start=df_features_final['Date'].iloc[-1] + pd.Timedelta(days=1), periods=14, freq='D')
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Sales': future_predictions})

    plt.figure()
    plt.plot(df_features_final['Date'].tail(90), df_features_final['Sales'].tail(90), label='Historical Sales')
    plt.plot(forecast_df['Date'], forecast_df['Forecasted Sales'], label='Forecasted Sales', color='red', linestyle='--')
    plt.title('Sales Forecast for the Next 14 Days')
    plt.ylabel('Sales ($)')
    plt.xlabel('Date')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print("\n--- Sales Forecast ---")
    print(forecast_df)
    print("\nScript finished successfully.")

if __name__ == '__main__':
    main()
