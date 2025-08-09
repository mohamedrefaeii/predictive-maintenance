#  Predictive Maintenance System using Time-Series Analysis

##  Overview

This project implements a **predictive maintenance system** for industrial equipment using **time-series sensor data**. The system predicts equipment failures or estimates Remaining Useful Life (RUL), helping reduce downtime and optimize maintenance schedules.

It integrates **machine learning**, **deep learning**, and **interactive dashboards** to deliver real-time insights. Suitable for industries like **manufacturing**, **energy**, and **transportation**.

>  This project showcases strong skills in time-series analysis, supervised learning, feature engineering, model evaluation, and data visualization — making it a solid addition to any AI/Data Science portfolio.

##  Key Features

- Time-series preprocessing & feature engineering  
- Failure classification or RUL prediction using:
  -  Tree-based models (XGBoost / Random Forest)
  -  LSTM deep learning model
- Performance evaluation using metrics like RMSE, Accuracy, AUC
- Visualizations (sensor trends, feature importance, confusion matrix)
- Interactive Streamlit app for real-time predictions
- Modular, reusable Python codebase
- Cloud-deployed web app for public interaction

##  Repository Structure

```
predictive-maintenance/
├── data/                     # Sample CSV data (e.g., engine_data.csv)
├── src/
│   ├── preprocessing.py      # Data cleaning, feature engineering
│   ├── model_training.py     # Model training & hyperparameter tuning
│   ├── inference.py          # Inference pipeline
│   ├── dashboard.py          # Streamlit app logic
│   ├── visualization.py      # Metrics & plots
├── notebooks/
│   └── pipeline.ipynb        # End-to-end Jupyter pipeline
├── plots/                    # Saved visualizations (e.g., loss curves)
├── requirements.txt          # Python dependencies
├── .gitignore                # Ignored files (e.g., large data, weights)
├── LICENSE                   # MIT License
└── README.md                 # Project documentation (this file)
```


###  Install Dependencies

```bash
git clone https://github.com/mohamedrefaeii/predictive-maintenance.git
cd predictive-maintenance
pip install -r requirements.txt
```

###  Run the App Locally

```bash
streamlit run src/dashboard.py
```

### 📤 Upload CSV


```csv
timestamp,temperature,pressure,vibration,...
0,35.6,1.02,0.003,...
1,35.8,1.03,0.002,...
...
```

> Sample available in `data/engine_data.csv`

##  Models Used

| Model Type      | Algorithm         | Task                      |
|----------------|-------------------|---------------------------|
| Tree-Based      | XGBoost, RandomForest | Classification / Regression |
| Deep Learning   | LSTM (TensorFlow) | Time-Series RUL Prediction |

Models are evaluated using:

-  Classification: Accuracy, Precision, Recall, F1, ROC-AUC
-  Regression: MAE, RMSE, R²

Hyperparameter tuning done via `GridSearchCV` and `Optuna`.

##  Visualizations

All plots are saved in the `plots/` folder:

- Sensor Trends  
- Confusion Matrix  
- ROC Curve  
- Loss Curves  
- Feature Importance  

##  Deployment


###  Local

```bash
streamlit run src/dashboard.py
```

##  Tools & Libraries

| Task                  | Library                         |
|-----------------------|----------------------------------|
| Data Processing       | pandas, numpy                   |
| ML Models             | scikit-learn, xgboost           |
| Deep Learning         | tensorflow or pytorch           |
| Visualization         | matplotlib, seaborn, plotly     |
| Web App               | streamlit                       |
| Hyperparameter Tuning | optuna, GridSearchCV            |

##  Recommended Setup

- Python 3.8+
- Works on standard laptops
- For LSTM training: use **Google Colab** with GPU

##  .gitignore Highlights

```bash
__pycache__/
*.pyc
.venv/
*.env
*.h5
data/*.csv
```

##  Industrial Relevance

Predictive Maintenance is a real-world application critical to sectors like:

-  Manufacturing (e.g., machines in Egyptian factories)
-  Energy (e.g., turbine failure prediction)
-  Transportation (e.g., engine monitoring)

##  Success Criteria

-  >85% classification accuracy or <10% RMSE
-  Fully functional Streamlit app with real-time feedback
-  Clear notebook with reproducible pipeline
-  Cloud deployment with working predictions
-  Ready for job interviews and GitHub showcasing


## 👨‍💻 Author

**Mohamed Refaei**  
[GitHub](https://github.com/mohamedrefaeii)  

