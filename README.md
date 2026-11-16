# Poultry ML App

An ERPNext/Frappe custom app that uses machine learning to predict **poultry shed construction costs**, with confidence scoring and cost-driver analysis, directly inside the ERP.

## What it does
- Predicts the estimated construction cost from shed details entered in ERPNext.
- Uses a trained **classification + regression model** (Scikit-learn).
- Shows a **confidence score** and the main **cost drivers** behind each prediction.
- Fills the prediction fields in the form automatically through a data pipeline between the model and the live app.

## Tech stack
Python, Scikit-learn, Pandas, NumPy, Frappe/ERPNext (branch `version-16`), MySQL, Git

## Project structure
| Path | Purpose |
|---|---|
| `poultry_ml_app/ml_models/` | Trained model files |
| `poultry_ml_app/predictor.py` | Loads the model and returns predictions |
| `poultry_ml_app/hooks.py` | Frappe hooks that connect the app to ERPNext |
| `poultry_ml_app/patches/` | Database patches |
| `poultry_ml_app/templates/` | App templates |

## Problems solved
- **Class imbalance:** The model was biased toward the majority class. Fixed using class-weighted training.
- **Version mismatch:** Scikit-learn versions differed between training and production. Fixed so the saved model loads reliably.
- **JSON serialization:** NumPy values failed to serialize in API responses. Converted them to native Python types.
- **Testing:** Verified predictions across 50+ records.

## Installation
```bash
cd ~/frappe-bench
bench get-app https://github.com/rohitshewale4165-eng/poultry_ml_app --branch version-16
bench --site [your-site-name] install-app poultry_ml_app
bench --site [your-site-name] migrate
```

## Screenshots
[Add 1-2 screenshots of the prediction form in ERPNext]

## Results
[Add your model's metric here, e.g. accuracy / MAE / R², only if you have the real number]

## Author
Rohit Shewale | [LinkedIn](https://www.linkedin.com/in/rohit-shewale-6b154a25b/) | [GitHub](https://github.com/rohitshewale4165-eng)
