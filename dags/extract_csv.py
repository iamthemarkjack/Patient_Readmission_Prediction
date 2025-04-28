from datetime import datetime
import os
import zipfile
import pandas as pd
import logging
from airflow import DAG
from airflow.operators.python import PythonOperator

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Setting up Logging
logging.basicConfig(
    filename='./logs/extract_csv.log',
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
CWD = os.getcwd()
RAW_DATA_PATH = os.path.join(CWD, 'data', 'raw', 'archive.zip')
PROCESSED_DATA_DIR = os.path.join(CWD, 'data', 'processed')
CSV_FILENAME = 'data.csv'

def extract_data(**kwargs):
    """
    Extract the CSV file from zip file and load it into a pandas DataFrame
    then save the CSV to the processed directory.
    """
    logger.info(f"Starting extraction of CSV from {RAW_DATA_PATH}")

    # Ensure the processed directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    logger.info(f"Ensured processed directory exists: {PROCESSED_DATA_DIR}")

    # Path to save the extracted csv
    extracted_csv_path = os.path.join(PROCESSED_DATA_DIR, CSV_FILENAME)

    try:
        # Extract the CSV from the zip file
        with zipfile.ZipFile(RAW_DATA_PATH, 'r') as zip_ref:
            # Get all file names in the zip
            file_list = zip_ref.namelist()
            logger.info(f"Files found in archive: {file_list}")
            
            # Extract the CSV file
            csv_found = False
            for file_name in file_list:
                if file_name.endswith('.csv'):
                    # Extract the file
                    logger.info(f"Extracting {file_name} from archive")
                    with zip_ref.open(file_name) as source, open(extracted_csv_path, 'wb') as target:
                        target.write(source.read())
                    logger.info(f"Successfully extracted {file_name} to {extracted_csv_path}")
                    csv_found = True
                    break
            
            if not csv_found:
                logger.error("No CSV file found in the archive")
                raise FileNotFoundError("No CSV file found in the archive")
    
    except zipfile.BadZipFile:
        logger.error(f"Failed to open {RAW_DATA_PATH} - not a valid zip file")
        raise
    except FileNotFoundError:
        logger.error(f"Archive file not found at {RAW_DATA_PATH}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during extraction: {str(e)}")
        raise

# Define the DAG
with DAG(
    'extract_csv_from_archive',
    default_args=default_args,
    description='Extract CSV from zip archive',
    schedule_interval=None,  # Set to None for manual triggering
    catchup=False,
) as dag:
    
    # Define the task to extract
    extract_task = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data,
        provide_context=True,
    )
    
    # Set task dependencies
    extract_data