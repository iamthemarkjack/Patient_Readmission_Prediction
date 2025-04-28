from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import numpy as np
import os
import subprocess
import logging

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Setting up Logging
logging.basicConfig(
    filename='./logs/data_processing.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    'owner' : 'ep21b030',
    'start_date' : datetime(2025, 4, 23),
    'retries' : 1,
}

# Define paths
INPUT_PATH = "data/processed/data.csv"
OUTPUT_PATH = "data/processed/data_featured.csv"

def read_data():
    """Read the CSV data from processed directory"""
    logger.info(f"Reading data from {INPUT_PATH}")
    if not os.path.exists(INPUT_PATH):
        logger.error(f"Data file not found at {INPUT_PATH}")
        raise FileNotFoundError(f"Data file not found at {INPUT_PATH}")
    
    df = pd.read_csv(INPUT_PATH)
    logger.info(f"Data loaded successfully with shape: {df.shape}")
    return df

#==========Helper functions to create new features=================

def add_age_group(df):
    """Function to add Age Group Feature"""
    bins = [18, 30, 45, 60, 75, 100]
    labels = ['18-30', '31-45', '46-60', '61-75', '76+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False, include_lowest=True)
    df['age_group'] = df['age_group'].cat.add_categories('Other').fillna('Other')

def add_diagnosis_group(df):
    """Function to Group Primary Diagnosis"""
    diagnosis_mapping = {
        'Hypertension': 'Cardiac',
        'Heart Attack': 'Cardiac',
        'Asthma': 'Respiratory',
        'COPD': 'Respiratory',
        'Diabetes': 'Diabetes-related'
    }
    df['diagnosis_group'] = df['primary_diagnosis'].map(diagnosis_mapping).fillna('Other')

def add_hospital_stay_features(df):
    """Function to create Hospital Stay Features"""
    stay_bins = [0, 3, 7, 14, np.inf]
    stay_labels = ['0-3 days', '4-7 days', '8-14 days', '>14 days']
    df['stay_bucket'] = pd.cut(df['days_in_hospital'], bins=stay_bins, labels=stay_labels)
    df['stay_bucket'] = df['stay_bucket'].cat.add_categories('Other').fillna('Other')

    df['long_stay'] = np.where(df['days_in_hospital'] > 7, 1, 0)
    df['long_stay'] = df['long_stay'].fillna(-1)

def add_procedure_features(df):
    """Function to create Procedure Features"""
    procedure_bins = [0, 1, 3, 5, np.inf]
    procedure_labels = ['None', '1-2', '3-5', '5+']
    df['procedure_category'] = pd.cut(df['num_procedures'], bins=procedure_bins, labels=procedure_labels)
    df['procedure_category'] = df['procedure_category'].cat.add_categories('Other').fillna('Other')

    df['procedures_diagnosis_interaction'] = df['num_procedures'] * df['diagnosis_group'].factorize()[0]

def add_comorbidity_features(df):
    """Function to create Comorbidity Features"""
    comorbidity_threshold = 3
    df['high_risk_comorbidity'] = np.where(df['comorbidity_score'] >= comorbidity_threshold, 1, 0)
    df['high_risk_comorbidity'] = df['high_risk_comorbidity'].fillna(-1)

def add_interaction_features(df):
    """Function to create Interaction Terms"""
    df['age_comorbidity_interaction'] = df['age'] * df['comorbidity_score']
    df['procedures_age_interaction'] = df['num_procedures'] * df['age']
    df['procedures_comorbidity_interaction'] = df['num_procedures'] * df['comorbidity_score']

#================End of Helper functions==============================

def add_features(df):
    """Master function to add all features"""
    try:
        add_age_group(df)
        logger.info("Age group features added.")
    except Exception as e:
        logger.error(f"Error while adding the age group features: {e}")
        raise
    try:
        add_diagnosis_group(df)
        logger.info("Diagonis group features added.")
    except Exception as e:
        logger.error(f"Error while adding the diagnosis group features: {e}")
        raise
    try:    
        add_hospital_stay_features(df)
        logger.info("Hospital stay features added.")
    except Exception as e:
        logger.error(f"Error while adding the hospital stay features: {e}")
        raise
    try:        
        add_procedure_features(df)
        logger.info("Procedure features added.")
    except Exception as e:
        logger.error(f"Error while adding the procedure features: {e}")
        raise
    try:        
        add_comorbidity_features(df)
        logger.info("Comorbidity features added.")
    except Exception as e:
        logger.error(f"Error while adding the acomorbidity features: {e}")
        raise
    try:
        add_interaction_features(df)
        logger.info("Interaction features added.")
    except Exception as e:
        logger.error(f"Error while adding the interaction features: {e}")
        raise

    return df

def save_data(df):
    """Save processed dataframe to output path"""
    logger.info(f"Saving data to {OUTPUT_PATH}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Save data
    try:
        df.to_csv(OUTPUT_PATH, index=False)
        logger.info(f"Data saved successfully to {OUTPUT_PATH}")
        return OUTPUT_PATH
    except Exception as e:
        logger.error(f"Error while saving the data: {e}")
        raise

# Define the DAG
with DAG(
    'feature_engineering',
    default_args=default_args,
    description='A DAG to add features to processed data',
    schedule_interval=None,  # Set to None for manual triggering
    catchup=False
) as dag:
    
    # Define tasks
    task_read_data = PythonOperator(
        task_id='read_data',
        python_callable=read_data,
        dag=dag
    )
    
    task_add_features = PythonOperator(
        task_id='add_features',
        python_callable=add_features,
        op_args=[task_read_data.output],
        dag=dag
    )
    
    task_save_data = PythonOperator(
        task_id='save_data',
        python_callable=save_data,
        op_args=[task_add_features.output],
        dag=dag
    )
    
    # Define dependencies
    task_read_data >> task_add_features >> task_save_data