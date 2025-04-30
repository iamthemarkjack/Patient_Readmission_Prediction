# DAG for processing the data - add features

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
import pandas as pd
import os
import logging

from utils.add_feature import add_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data_processing_dag.log")
    ]
)
logger = logging.getLogger("Feature Engineering with DAG")

# Default arguments for the DAG
default_args = {
    'owner': 'ep21b030',
    'start_date': datetime(2025, 4, 23),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

# Define paths
CWD = os.getcwd()
INPUT_PATH = os.path.join(CWD, "data/processed/raw_data.csv")
OUTPUT_PATH = os.path.join(CWD, "data/processed/processed_data.csv")

def read_data():
    """Read the CSV data from processed directory"""
    logger.info(f"Reading data from {INPUT_PATH}")
    if not os.path.exists(INPUT_PATH):
        logger.error(f"Data file not found at {INPUT_PATH}")
        raise FileNotFoundError(f"Data file not found at {INPUT_PATH}")
    
    df = pd.read_csv(INPUT_PATH)
    logger.info(f"Data loaded successfully with shape: {df.shape}")
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

def delete_input_file():
    """Delete the input CSV file after processing"""
    try:
        if os.path.exists(INPUT_PATH):
            os.remove(INPUT_PATH)
            logger.info(f"Successfully deleted input file: {INPUT_PATH}")
        else:
            logger.warning(f"Input file not found for deletion: {INPUT_PATH}")
    except Exception as e:
        logger.error(f"Error while deleting input file: {e}")
        raise

def process_data(**kwargs):
    """
    Process data end-to-end and delete input file
    """
    # Read data
    logger.info("Starting data processing pipeline")
    df = read_data()
    
    # Add features
    df = add_features(df)
    
    # Save data
    save_data(df)
    
    # Delete input file
    delete_input_file()
    
    logger.info("Data processing completed successfully")

# Define the DAG
with DAG(
    'feature_engineering',
    default_args=default_args,
    description='A DAG to add features to processed data and delete input file',
    schedule_interval='*/30 * * * *',  # Check every 30 minutes
    catchup=False
) as dag:
    
    # File sensor to wait for raw_data.csv
    file_sensor = FileSensor(
        task_id='wait_for_raw_data',
        filepath=INPUT_PATH,
        fs_conn_id='fs_default',
        poke_interval=1800,  # Check every 30 minutes (in seconds)
        timeout=60 * 60 * 24,  # Timeout after 24 hours
        mode='poke',
        soft_fail=True,
    )
    
    # Process data
    process_task = PythonOperator(
        task_id='process_data',
        python_callable=process_data,
        provide_context=True,
    )
    
    # Set task dependencies
    file_sensor >> process_task