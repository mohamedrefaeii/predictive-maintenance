Predictive Maintenance for Industrial Equipment
Overview
This project implements a machine learning system for predictive maintenance of industrial equipment, utilizing time-series sensor data to forecast equipment failures or estimate remaining useful life (RUL). The system integrates data preprocessing, feature engineering, predictive modeling, and interactive visualizations, demonstrating core machine learning and data science competencies. It includes a Streamlit-based web application for real-time predictions and a Jupyter notebook for a detailed end-to-end pipeline, making it a robust addition to a data science portfolio.
Features

Data Preprocessing: Cleans and normalizes time-series sensor data, with feature engineering for temporal patterns (e.g., rolling averages, lag features).
Predictive Models: Employs supervised learning models, including Random Forest, XGBoost (for classification or regression), and LSTM (for time-series prediction).
Model Evaluation: Assesses performance using metrics such as accuracy, precision, recall, F1-score (classification), and RMSE (regression).
Interactive Dashboard: Provides a Streamlit web application for users to upload sensor data, view failure predictions, and explore visualizations.
Visualizations: Includes time-series plots, feature importance charts, and performance metrics using Plotly and Matplotlib.
Reproducible Workflow: Offers a Jupyter notebook (notebooks/pipeline.ipynb) detailing the end-to-end machine learning pipeline.

Relevance
Predictive maintenance is critical for industries such as manufacturing, energy, and transportation, particularly in Egypt’s growing industrial sector. This project showcases proficiency in time-series analysis, feature engineering, model development, and deployment, aligning with the demands of machine learning and data science roles.
Installation
To set up the project locally, follow these steps:

Clone the Repository:
git clone https://github.com/mohamedrefaeii/predictive-maintenance.git
cd predictive-maintenance


Install Dependencies:Ensure Python 3.8+ is installed. Install required packages using:
pip install -r requirements.txt


Download Sample Data (optional):The repository includes a sample dataset (data/engine_data.csv). Alternatively, download the full NASA Turbofan Engine Degradation Dataset or Kaggle Predictive Maintenance Dataset and place a subset in the data/ directory.


Usage

Run the Streamlit Dashboard:Launch the interactive web application:
streamlit run src/dashboard.py

Access the app at http://localhost:8501. Upload a CSV file with sensor data to view predictions and visualizations.

Explore the Jupyter Notebook:Open notebooks/pipeline.ipynb in Jupyter to review the end-to-end pipeline, including data preprocessing, model training, and evaluation.

Run Predictions Locally:Use the inference script to generate predictions:
python src/inference.py --input data/engine_data.csv



Dependencies
The project relies on the following Python libraries (listed in requirements.txt):

pandas: Data manipulation and preprocessing
numpy: Numerical computations
scikit-learn: Machine learning models (Random Forest)
xgboost: Gradient boosting model
tensorflow: Deep learning (LSTM)
plotly: Interactive visualizations
matplotlib: Static plots
seaborn: Enhanced visualizations
streamlit: Web application framework
optuna: Hyperparameter tuning

Install all dependencies with:
pip install -r requirements.txt

Project Structure
predictive-maintenance/
├── data/                     # Sample sensor data (e.g., engine_data.csv)
├── src/                      # Source code
│   ├── preprocessing.py      # Data cleaning and feature engineering
│   ├── model_training.py     # Model training and tuning
│   ├── inference.py          # Prediction pipeline
│   ├── dashboard.py          # Streamlit web app
│   ├── visualization.py      # Visualization functions
├── notebooks/                # Jupyter notebooks
│   ├── pipeline.ipynb        # End-to-end pipeline
├── plots/                    # Saved visualizations (e.g., loss_curve.png)
├── screenshots/              # Dashboard screenshots
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
├── .gitignore                # Ignored files
└── LICENSE                   # MIT License

Example
To predict equipment failure risk:

Place a CSV file (e.g., data/engine_data.csv) with sensor readings (columns: timestamp, sensor1, sensor2, etc.) in the data/ directory.
Run the Streamlit app:streamlit run src/dashboard.py


Upload the CSV file via the web interface.
View the output, e.g., “Failure Probability: 85%” with a time-series plot of predictions.

Example output from notebooks/pipeline.ipynb:

Classification metrics: Accuracy: 0.88, F1-score: 0.85
Regression metrics: RMSE: 12.3, R²: 0.90
Visualization: Feature importance plot showing sensor1 as the top predictor.

Screenshots

Deployed App

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Dataset: NASA Turbofan Engine Degradation Dataset (public domain).
Libraries: Built with open-source tools like scikit-learn, TensorFlow, and Streamlit.
Inspiration: Addressing predictive maintenance needs in industrial settings, with relevance to Egypt’s manufacturing and energy sectors.

Contact
For questions or contributions, contact Mohamed Refaei via GitHub (mohamedrefaeii) or email (mohameddrefaee6@gmail.com).
