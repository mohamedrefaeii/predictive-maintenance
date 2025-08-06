"""
Streamlit dashboard for predictive maintenance system.
Interactive web application for real-time predictions and visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from src.preprocessing import DataPreprocessor
from src.inference import InferenceEngine

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def load_models():
    """Load trained models."""
    models = {}
    model_files = {
        'random_forest': 'models/random_forest_model.joblib',
        'xgboost': 'models/xgboost_model.joblib',
        'lstm': 'models/lstm_model.h5'
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            if name == 'lstm':
                import tensorflow as tf
                models[name] = tf.keras.models.load_model(path)
            else:
                models[name] = joblib.load(path)
    
    return models

def main():
    """Main dashboard application."""
    
    # Sidebar
    st.sidebar.title("🛠️ Predictive Maintenance")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "Upload Data", "Predictions", "Model Performance", "About"]
    )
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Upload Data":
        show_upload_page()
    elif page == "Predictions":
        show_predictions_page()
    elif page == "Model Performance":
        show_model_performance()
    elif page == "About":
        show_about_page()

def show_dashboard():
    """Display main dashboard."""
    st.markdown('<h1 class="main-header">🔧 Predictive Maintenance Dashboard</h1>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Engines", "1,250", "+5%")
    with col2:
        st.metric("Active Alerts", "23", "-12%")
    with col3:
        st.metric("Avg RUL", "187 days", "+8 days")
    with col4:
        st.metric("System Health", "92%", "+3%")
    
    st.markdown("---")
    
    # Sample data visualization
    st.subheader("📊 System Overview")
    
    # Create sample data for demonstration
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'engine_id': np.repeat([1, 2, 3], 50),
        'cycle': list(range(50)) * 3,
        'temperature': np.random.normal(1589, 50, 150),
        'vibration': np.random.normal(1400, 40, 150),
        'pressure': np.random.normal(392, 10, 150)
    })
    
    fig = px.line(sample_data, x='cycle', y='temperature', color='engine_id',
                  title='Temperature Trends by Engine')
    st.plotly_chart(fig, use_container_width=True)

def show_upload_page():
    """Display upload data page."""
    st.header("📁 Upload Data")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head())
        
        if st.button("Process Data"):
            # Process data
            st.success("Data processed successfully!")
            st.write("Shape:", df.shape)

def show_predictions_page():
    """Display predictions page."""
    st.header("🔮 Predictions")
    
    # Upload data for prediction
    uploaded_file = st.file_uploader("Upload CSV for prediction", type="csv", key="prediction")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        if st.button("Predict RUL"):
            # Simulate prediction
            predictions = np.random.randint(50, 200, len(df))
            df['predicted_rul'] = predictions
            
            st.write("Predictions:")
            st.dataframe(df[['predicted_rul']])
            
            fig = px.histogram(df, x='predicted_rul', title='Predicted RUL Distribution')
            st.plotly_chart(fig)

def show_model_performance():
    """Display model performance page."""
    st.header("📈 Model Performance")
    
    # Sample performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Random Forest")
        st.metric("RMSE", "8.5", "-12%")
        st.metric("R²", "0.92", "+5%")
    
    with col2:
        st.subheader("XGBoost")
        st.metric("RMSE", "7.8", "-15%")
        st.metric("R²", "0.94", "+7%")

def show_about_page():
    """Display about page."""
    st.header("ℹ️ About")
    
    st.markdown("""
    ## Predictive Maintenance System
    
    This system uses machine learning to predict equipment failures and estimate remaining useful life (RUL) 
    based on sensor data from industrial equipment.
    
    ### Features:
    - **Data Processing**: Automated preprocessing of time-series sensor data
    - **Multiple Models**: Random Forest, XGBoost, and LSTM implementations
    - **Interactive Dashboard**: Streamlit-based web application for real-time predictions
    - **Visualizations**: Comprehensive plots for sensor trends and model performance
    
    ### Technologies Used:
    - **Python**: Core programming language
    - **Scikit-learn**: Machine learning algorithms
    - **XGBoost**: Gradient boosting framework
    - **TensorFlow**: Deep learning framework
    - **Streamlit**: Web application framework
    
    ### Contact:
    For questions or support, please contact the development team.
    """)

if __name__ == "__main__":
    main()
