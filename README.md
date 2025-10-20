# 📈 Predictive Sales Forecaster using XGBoost

An advanced, object-oriented machine learning pipeline designed to provide **rural and small businesses** with highly accurate, data-driven sales predictions. It leverages the power of **XGBoost** to transform raw historical data into **actionable inventory, staffing, and marketing strategies**.



## ✨ Why XGBoost? The Predictive Engine

We use a high-performance **XGBoost** model. Think of it as a **"team of experts"** (*Gradient Boosting*) where each new expert's sole job is to correct the mistakes of the previous one. This powerful, iterative approach captures complex patterns, like seasonality and local event impact, with incredible accuracy.

## 🎯 Core Features & Actionable Insights

| Feature Category | Description | Business Value |
| :--- | :--- | :--- |
| **Advanced Feature Engineering** | Automatically creates powerful features from time-series data: **Lag Features** (e.g., yesterday's sales), **Rolling Window Averages**, and **Cyclical Features** (sin/cos transformations for months/weekdays to capture seasonality). | Turns dates into data the model understands, leading to highly predictive signals. |
| **Intelligent Tuning** | Uses **GridSearchCV** to automatically find the optimal settings (hyperparameters) for the XGBoost model specific to your data, ensuring maximum accuracy and reliability. | Guarantees the best possible performance without manual, time-consuming effort. |
| **Robust Training** | Implements **Early Stopping** to prevent overfitting on the training data and reduce computational time. | Ensures the model generalizes well to future, unseen sales data. |
| **Actionable Insights** | Generates a **Feature Importance** chart showing the biggest drivers of sales (e.g., weekends, holidays). Provides clear metrics: **MAE** (average prediction error in dollars) and **R²** (variance explained). | The model tells you *why* a prediction was made, enabling smarter decision-making. |
| **Confidence** | Produces a final forecast with a **95% Confidence Interval**, providing a realistic range for expected outcomes. | Allows for robust risk assessment and inventory buffer planning. |

---

## 🛠️ Tech Stack & File Description

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Core Engine** | **XGBoost** | High-performance Gradient Boosting implementation. |
| **ML Toolkit** | `scikit-learn` | Model selection (`GridSearchCV`) and evaluation. |
| **Data** | `pandas`, `NumPy` | Manipulation of time-series and numerical data. |
| **Visualization** | `Matplotlib`, `Seaborn` | Generating the multi-panel **Evaluation Dashboard** and **Final Forecast Plot**. |

### File: `Advanced_Sales_Forecaster.py`

This **object-oriented Python script** encapsulates the entire end-to-end sales forecasting pipeline within a `SalesForecaster` class. It manages data simulation, sophisticated feature engineering, hyperparameter tuning, model training, and the final generation of the evaluation dashboard and 14-day sales forecast with confidence intervals.

---

## ⚡ Getting Started

Follow these steps to run the project on your local machine.

### 1. Prerequisites

Ensure you have **Python 3.7+** installed.

### 2. Clone the Repository

```bash
git clone [https://github.com/ankitscse27/Predictive-Analytics-for-Rural-Business-Growth.git](https://github.com/ankitscse27/Predictive-Analytics-for-Rural-Business-Growth.git)
cd Predictive-Analytics-for-Rural-Business-Growth
3. Set Up a Virtual Environment (Recommended)
Bash

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
4. Install Dependencies
Create a requirements.txt file with the following, then install:

Plaintext

pandas
numpy
xgboost
scikit-learn
matplotlib
seaborn
Bash

pip install -r requirements.txt
5. Run the Script
The script will print progress, display the Model Evaluation Dashboard, and show the final 14-Day Forecast plot.

Bash

python Advanced_Sales_Forecaster.py
👤 Author & License
GitHub: @ankitscse27

License: This project is licensed under the MIT License.
