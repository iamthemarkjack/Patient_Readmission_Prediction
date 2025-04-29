import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import logging
import os
import json
import time
import traceback
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("streamlit.log")
    ]
)
logger = logging.getLogger("readmission_predictor")

# Set page config
st.set_page_config(
    page_title="Patient Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint configuration - use environment variable if available
API_URL = os.environ.get("API_URL", "http://backend:8000")

# Add session state for tracking runtime metrics
if 'api_response_times' not in st.session_state:
    st.session_state.api_response_times = []
if 'prediction_count' not in st.session_state:
    st.session_state.prediction_count = 0
if 'error_count' not in st.session_state:
    st.session_state.error_count = 0
if 'model_info' not in st.session_state:
    st.session_state.model_info = {}

def get_model_info() -> Dict:
    """Get model information from the API."""
    url = f"{API_URL}/version"
    logger.info(f"Getting model information from {url}")
    
    try:
        start_time = time.time()
        response = requests.get(url, timeout=10)
        response_time = time.time() - start_time
        
        st.session_state.api_response_times.append(response_time)
        
        if response.status_code == 200:
            logger.info(f"Model info request successful. Response time: {response_time:.2f}s")
            model_info = response.json()
            st.session_state.model_info = model_info
            return model_info
        else:
            logger.error(f"Model info request failed with status code {response.status_code}")
            st.session_state.error_count += 1
            return {"model_name": "unknown", "model_version": "unknown", "model_type": "unknown"}
    
    except requests.exceptions.RequestException as e:
        logger.error(f"API connection error: {str(e)}")
        st.session_state.error_count += 1
        return {"model_name": "unknown", "model_version": "unknown", "model_type": "unknown", "error": str(e)}

def get_model_status() -> Dict:
    """Get model server status from the API."""
    url = f"{API_URL}/health"
    logger.info(f"Checking model status at {url}")
    
    try:
        start_time = time.time()
        response = requests.get(url, timeout=10)
        response_time = time.time() - start_time
        
        st.session_state.api_response_times.append(response_time)
        
        if response.status_code == 200:
            logger.info(f"Model status check successful. Response time: {response_time:.2f}s")
            status_data = response.json()
            
            # Update model info if available
            if 'model_name' in status_data or 'model_version' in status_data or 'model_type' in status_data:
                st.session_state.model_info.update({
                    'model_name': status_data.get('model_name', st.session_state.model_info.get('model_name', 'unknown')),
                    'model_version': status_data.get('model_version', st.session_state.model_info.get('model_version', 'unknown')),
                    'model_type': status_data.get('model_type', st.session_state.model_info.get('model_type', 'unknown')),
                    'f1_score': status_data.get('f1_score', st.session_state.model_info.get('f1_score', None))
                })
            
            return status_data
        else:
            logger.error(f"Model status check failed with status code {response.status_code}")
            st.session_state.error_count += 1
            return {"model_status": "unavailable", "status": "error"}
    
    except requests.exceptions.RequestException as e:
        logger.error(f"API connection error: {str(e)}")
        st.session_state.error_count += 1
        return {"model_status": "unavailable", "status": "error", "error": str(e)}

def predict_patient(patient_data: Dict) -> Dict:
    """Send a single patient prediction request to the API."""
    logger.info(f"Predicting readmission for patient with diagnosis: {patient_data.get('primary_diagnosis', 'unknown')}")
    
    try:
        # Convert any string numbers to proper numeric types
        if isinstance(patient_data.get('comorbidity_score'), str) and patient_data.get('comorbidity_score').isdigit():
            patient_data['comorbidity_score'] = int(patient_data['comorbidity_score'])
            
        if isinstance(patient_data.get('age'), str) and patient_data.get('age').isdigit():
            patient_data['age'] = int(patient_data['age'])
            
        api_fields = {
            "age": patient_data.get("age"),
            "gender": patient_data.get("gender"),
            "primary_diagnosis": patient_data.get("primary_diagnosis"),
            "num_procedures": patient_data.get("num_procedures"),
            "days_in_hospital": patient_data.get("days_in_hospital"),
            "comorbidity_score": patient_data.get("comorbidity_score"),
            "discharge_to": patient_data.get("discharge_to")
        }
             
        logger.info(f"Sending prediction request with data: {json.dumps(api_fields)}")
        
        # Make the API request with the cleaned data
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/predict", 
            json=api_fields,
            timeout=30
        )
        response_time = time.time() - start_time
        
        st.session_state.api_response_times.append(response_time)
        st.session_state.prediction_count += 1
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Prediction successful. Will be readmitted: {result.get('will_be_readmitted', 'unknown')}. Response time: {response_time:.2f}s")
            
            # Update model info if new info is available
            if 'model_version' in result and result['model_version'] != 'unknown':
                if 'model_info' not in st.session_state:
                    st.session_state.model_info = {}
                st.session_state.model_info['model_version'] = result.get('model_version')
                if 'model_type' in result:
                    st.session_state.model_info['model_type'] = result.get('model_type')
            
            return result
        else:
            error_msg = f"API error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            st.session_state.error_count += 1
            return {"error": error_msg}
    
    except Exception as e:
        logger.error(f"Error in predict_patient: {str(e)}")
        logger.error(traceback.format_exc())
        st.session_state.error_count += 1
        return {"error": str(e)}

def predict_patients_batch(patients_data: List[Dict]) -> List[Dict]:
    """Send a batch prediction request to the API."""
    logger.info(f"Processing batch prediction for {len(patients_data)} patients")
    
    try:
        # Clean patient data to match API expectations
        cleaned_patients = []
        for patient in patients_data:
            # Convert any string numbers to proper numeric types
            if isinstance(patient.get('comorbidity_score'), str) and patient.get('comorbidity_score').isdigit():
                patient['comorbidity_score'] = int(patient['comorbidity_score'])
            if isinstance(patient.get('age'), str) and patient.get('age').isdigit():
                patient['age'] = int(patient['age'])
                
            # Use only the required fields for the API
            api_fields = {
                "age": patient.get("age"),
                "gender": patient.get("gender"),
                "primary_diagnosis": patient.get("primary_diagnosis"),
                "num_procedures": patient.get("num_procedures"),
                "days_in_hospital": patient.get("days_in_hospital"),
                "comorbidity_score": patient.get("comorbidity_score"),
                "discharge_to": patient.get("discharge_to")
            }
            cleaned_patients.append(api_fields)
        
        logger.info(f"Sending batch prediction request with {len(cleaned_patients)} records")
        
        # Make the API request with cleaned data
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/predict/batch", 
            json=cleaned_patients,
            timeout=60
        )
        response_time = time.time() - start_time
        
        st.session_state.api_response_times.append(response_time)
        st.session_state.prediction_count += len(patients_data)
        
        if response.status_code == 200:
            results = response.json()
            logger.info(f"Batch prediction successful for {len(results)} patients. Response time: {response_time:.2f}s")
            
            # Update model info if available in first result
            if results and 'model_version' in results[0] and results[0]['model_version'] != 'unknown':
                if 'model_info' not in st.session_state:
                    st.session_state.model_info = {}
                st.session_state.model_info['model_version'] = results[0].get('model_version')
                if 'model_type' in results[0]:
                    st.session_state.model_info['model_type'] = results[0].get('model_type')
            
            return results
        else:
            error_msg = f"API error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            st.session_state.error_count += 1
            return [{"error": error_msg}]
    
    except Exception as e:
        logger.error(f"Error in predict_patients_batch: {str(e)}")
        logger.error(traceback.format_exc())
        st.session_state.error_count += 1
        return [{"error": str(e)}]

#===============UI Components===============================

def render_header():
    """Render the application header."""
    st.markdown("""
    <div style='background-color:#0066cc;padding:10px;border-radius:10px'>
    <h1 style='color:white;text-align:center'>Patient Readmission Predictor</h1>
    </div>
    """, unsafe_allow_html=True)

def render_home_page():
    """Render the home page content."""
    st.write("## Welcome to the Patient Readmission Prediction System")
    st.markdown("""
     This application helps healthcare providers predict the likelihood of a patient being readmitted
     within 30 days after discharge.
     
     ### Key Features:
     - Predict readmission risk for individual patients
     - Perform batch predictions for multiple patients
     - Understand feature importance
     
     ### How to use:
     1. Navigate to "Single Prediction" to predict for an individual patient
     2. Navigate to "Batch Prediction" to upload and predict for multiple patients
     3. View "Model Information" to understand model performance and details
     """)
    
    # Display current model info
    try:
        # First try to get more detailed model info
        if not st.session_state.model_info:
            get_model_info()
            
        # Then get the current status
        model_status = get_model_status()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model Status")
            status_color = "green" if model_status.get('model_status', model_status.get('status')) == "healthy" else "red"
            st.markdown(f"<p style='color:{status_color};font-size:20px;'>Status: {model_status.get('model_status', model_status.get('status', 'unknown'))}</p>", unsafe_allow_html=True)
            
            # Display model information from session state
            model_info = st.session_state.model_info
            st.markdown(f"<p>Model name: {model_info.get('model_name', 'unknown')}</p>", unsafe_allow_html=True)
            st.markdown(f"<p>Model version: {model_info.get('model_version', 'unknown')}</p>", unsafe_allow_html=True)
            st.markdown(f"<p>Model type: {model_info.get('model_type', 'unknown')}</p>", unsafe_allow_html=True)
            
            if model_info.get('f1_score'):
                st.markdown(f"<p>F1 Score: {model_info.get('f1_score', 'unknown')}</p>", unsafe_allow_html=True)
        
        with col2:
            st.subheader("Application Metrics")
            st.metric("Total Predictions Made", st.session_state.prediction_count)
            
            if st.session_state.api_response_times:
                avg_response_time = sum(st.session_state.api_response_times) / len(st.session_state.api_response_times)
                st.metric("Average API Response Time", f"{avg_response_time:.2f}s")
            
            st.metric("API Errors", st.session_state.error_count)
    
    except Exception as e:
        st.error(f"Could not connect to the prediction API. Please check if the backend service is running.")
        logger.error(f"Error connecting to API: {str(e)}")

def render_single_prediction_page():
    """Render the single prediction page content."""
    st.write("## Predict Readmission Risk for a Single Patient")
    
    # Create form for patient data
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 100, 65)
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            primary_diagnosis = st.selectbox(
                "Primary Diagnosis",
                ["Diabetes", "COPD", "Heart Disease", "Kidney Disease", "Hypertension", 
                 "Heart Attack", "Asthma", "Other"]
            )
            days_in_hospital = st.slider("Days in Hospital", 1, 30, 3)
        
        with col2:
            num_procedures = st.slider("Number of Procedures", 0, 10, 1)
            comorbidity_score = st.slider("Comorbidity Score", 0, 5, 2)
            discharge_to = st.selectbox(
                "Discharged to",
                ["Home", "Home Health Care", "Skilled Nursing Facility",
                 "Rehabilitation Center", "Another Hospital", "Other"]
            )
        
        submit_button = st.form_submit_button(label="Predict Readmission Risk")
    
    if submit_button:
        patient_data = {
            "age": age,
            "gender": gender,
            "primary_diagnosis": primary_diagnosis,
            "days_in_hospital": days_in_hospital,
            "num_procedures": num_procedures,
            "comorbidity_score": comorbidity_score,
            "discharge_to": discharge_to
        }
        
        try:
            with st.spinner("Making prediction..."):
                result = predict_patient(patient_data)
                
                if "error" in result:
                    st.error(f"Error making prediction: {result['error']}")
                    return
            
            # Create columns for results
            col1, col2 = st.columns(2)
            
            # Display prediction
            with col1:
                st.subheader("Prediction Result")
                risk_percentage = result["readmission_probability"] * 100
                
                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_percentage,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Readmission Risk"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "green"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"},
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                st.plotly_chart(fig)
                
                if result["will_be_readmitted"]:
                    st.error("This patient is likely to be readmitted within 30 days.")
                else:
                    st.success("This patient is not likely to be readmitted within 30 days.")
            
            with col2:
                st.subheader("Patient Details")
                st.json(patient_data)
                
                st.subheader("Model Information")
                model_type = result.get('model_type', st.session_state.model_info.get('model_type', 'unknown'))
                model_version = result.get('model_version', st.session_state.model_info.get('model_version', 'unknown'))
                
                st.info(f"Prediction made using {model_type} model (version: {model_version})")
                
                # Calculate and display risk factors based on patient data
                st.subheader("Risk Factors")
                risk_factors = []
                
                if age > 75:
                    risk_factors.append("Advanced age (75+)")
                if int(comorbidity_score) >= 3:
                    risk_factors.append("High comorbidity score")
                if days_in_hospital > 7:
                    risk_factors.append("Extended hospital stay")
                if primary_diagnosis in ["Heart Disease", "Heart Attack", "COPD"]:
                    risk_factors.append(f"High-risk diagnosis: {primary_diagnosis}")
                if num_procedures >= 5:
                    risk_factors.append("Multiple procedures")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.warning(factor)
                else:
                    st.success("No major risk factors identified")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            logger.error(f"Error in single prediction: {str(e)}")
            logger.error(traceback.format_exc())

def render_batch_prediction_page():
    """Render the batch prediction page content."""
    st.write("## Predict Readmission Risk for Multiple Patients")
    
    # File upload option
    uploaded_file = st.file_uploader("Upload Patient Data CSV", type=["csv"])
    
    if uploaded_file is not None:
        # Read and display the data
        try:
            data = pd.read_csv(uploaded_file)
            logger.info(f"Uploaded file with {len(data)} rows and {len(data.columns)} columns")
            
            st.write("### Preview of uploaded data:")
            st.dataframe(data.head())
            
            # Check if required columns exist
            required_cols = ["age", "gender", "primary_diagnosis", "num_procedures",
                             "days_in_hospital", "comorbidity_score", "discharge_to"]
            
            # Check column names and suggest corrections for common variations
            column_mapping = {
                "time_in_hospital": "days_in_hospital",
                "discharge_to": "discharge_to",
                "primary_diag": "primary_diagnosis",
                "diagnosis": "primary_diagnosis",
                "discharged_to": "discharge_to"
            }
            
            # Apply mappings if needed
            renamed_columns = []
            for old_col, new_col in column_mapping.items():
                if old_col in data.columns and new_col not in data.columns:
                    data = data.rename(columns={old_col: new_col})
                    renamed_columns.append(f"'{old_col}' to '{new_col}'")
            
            if renamed_columns:
                st.info(f"Renamed columns: {', '.join(renamed_columns)}")
                logger.info(f"Renamed columns in uploaded data: {renamed_columns}")
            
            # Check for missing columns again after potential renaming
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                st.error(f"Missing required columns for prediction: {missing_cols}")
                return

            # Make predictions
            with st.spinner("Making batch predictions..."):
                results = predict_patients_batch(data.to_dict(orient='records'))

            if results and "error" not in results[0]:
                # Combine predictions with original data
                predictions = pd.DataFrame(results)
                combined_df = pd.concat([data.reset_index(drop=True), predictions], axis=1)

                # Show prediction summary
                st.write("### Prediction Results")
                st.dataframe(combined_df)

                # Show risk distribution chart
                if 'readmission_probability' in predictions.columns:
                    fig = px.histogram(
                        predictions,
                        x="readmission_probability",
                        nbins=10,
                        title="Distribution of Readmission Probabilities",
                        labels={"readmission_probability": "Probability of Readmission"}
                    )
                    st.plotly_chart(fig)
            else:
                st.error(f"Error during batch prediction: {results[0].get('error', 'Unknown error')}")
        except Exception as e:
            st.error(f"Failed to process the uploaded file: {str(e)}")
            logger.error(f"Error in render_batch_prediction_page: {str(e)}")
            logger.error(traceback.format_exc())

def main():
    render_header()

    menu = ["Home", "Single Prediction", "Batch Prediction"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "Home":
        render_home_page()
    elif choice == "Single Prediction":
        render_single_prediction_page()
    elif choice == "Batch Prediction":
        render_batch_prediction_page()

if __name__ == '__main__':
    main()