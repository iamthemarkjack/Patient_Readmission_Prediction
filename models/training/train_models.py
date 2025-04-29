# Import the libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from imblearn.over_sampling import SMOTENC

import pickle
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import dvc.api
import os
import json
import logging
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("model_building.log")
    ]
)
logger = logging.getLogger("MLFlow Experimentation")


def load_config(config_path="../config/model_params.yml"):
    """Load model configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

def load_data(file_path, repo_path, version):
    """Load data from DVC storage"""
    try:
        # URL of the data
        data_url = dvc.api.get_url(
            path=file_path,
            repo=repo_path,
            rev=version
        )
        logger.info(f"Loading data from {data_url} (version: {version})")
        return pd.read_csv(data_url)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(df):
    """Preprocess the data for model training"""
    logger.info("Starting data preprocessing")
    
    # Define the target and features
    X = df.drop(columns=['readmitted'])  # Features
    y = df['readmitted']  # Target

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    logger.info(f"Identified {len(categorical_cols)} categorical columns and {len(numerical_cols)} numerical columns")

    # Create label encoders for categorical columns
    encoders = {}
    for col in categorical_cols:
        e = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[col] = e.fit_transform(X[[col]])
        encoders[col] = e
    
    # Create and fit StandardScaler for numerical columns
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    logger.info(f"Applied StandardScaler to {len(numerical_cols)} numerical columns")

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")

    # Apply SMOTENC to balance the data (for categorical + numerical data)
    categorical_indices = [X.columns.get_loc(col) for col in categorical_cols]
    smote = SMOTENC(categorical_features=categorical_indices, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    logger.info(f"Applied SMOTENC: {X_train_smote.shape[0]} samples after resampling")
    
    result = {
        'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
        'X_train_smote': X_train_smote, 'y_train_smote': y_train_smote,
        'categorical_cols': categorical_cols, 'numerical_cols': numerical_cols,
        'encoders': encoders, 'scaler': scaler
    }
    
    return result

def train_and_evaluate_models(data, model_configs):
    """Train and evaluate multiple models using GridSearchCV"""
    X_train_smote = data['X_train_smote']
    y_train_smote = data['y_train_smote']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Dictionary to store best models and their performance
    best_models = {}
    best_metrics = {}
    
    # Initialize MLflow client for model registry operations
    client = MlflowClient()
    
    # Start parent MLflow run
    with mlflow.start_run(run_name="Model Comparison") as parent_run:
        parent_run_id = parent_run.info.run_id
        logger.info(f"Started parent MLflow run: {parent_run_id}")
        
        # Loop through each model
        for model_name, model_info in model_configs.items():
            logger.info(f"Training {model_name}...")
            
            # Get model class
            if model_name == 'RandomForest':
                model_class = RandomForestClassifier(random_state=42)
            elif model_name == 'XGBoost':
                model_class = XGBClassifier(random_state=42)
            elif model_name == 'HistGradientBoostingClassifier':
                model_class = HistGradientBoostingClassifier(random_state=42)
            else:
                logger.warning(f"Unknown model: {model_name}, skipping")
                continue
            
            with mlflow.start_run(run_name=f"{model_name} Tuning", nested=True) as model_run:
                # Set up GridSearchCV
                grid_search = GridSearchCV(
                    estimator=model_class,
                    param_grid=model_info['param_grid'],
                    cv=5,
                    n_jobs=-1,
                    verbose=2,
                    scoring='f1'  # Optimizing for F1 score
                )
                
                # Fit GridSearchCV
                logger.info(f"Starting GridSearchCV for {model_name}")
                grid_search.fit(X_train_smote, y_train_smote)
                logger.info(f"Completed GridSearchCV for {model_name}")
                
                # Log the parameters and results for each combination
                for i, params in enumerate(grid_search.cv_results_['params']):
                    with mlflow.start_run(run_name=f"{model_name}-Combination{i+1}", nested=True) as child_run:
                        mlflow.log_params(params)
                        mlflow.log_metric("mean_test_score", grid_search.cv_results_['mean_test_score'][i])
                
                # Get best model
                best_model = grid_search.best_estimator_
                
                # Make predictions on test set
                y_pred = best_model.predict(X_test)
                
                # Calculate metrics
                acc = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                # Store best model and metrics
                best_models[model_name] = best_model
                best_metrics[model_name] = {
                    'accuracy': acc,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
                
                # Log best parameters
                logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
                mlflow.log_params(grid_search.best_params_)
                
                # Log metrics
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1", f1)
                
                # Log model
                sign = infer_signature(X_test, y_pred)
                mlflow.sklearn.log_model(best_model, f"Best_{model_name}_Model", signature=sign)
                
                # Log metrics
                logger.info(f"{model_name} Metrics - Accuracy: {acc:.4f}, Precision: {precision:.4f}, "
                           f"Recall: {recall:.4f}, F1-score: {f1:.4f}")
        
        return best_models, best_metrics, client, parent_run_id

def select_and_register_best_model(best_models, best_metrics, client, data, model_dir):
    """Select the best model and register it in MLflow Model Registry"""
    # Find the best model based on F1 score
    best_model_name = max(best_metrics, key=lambda k: best_metrics[k]['f1'])
    final_best_model = best_models[best_model_name]
    final_metrics = best_metrics[best_model_name]
    
    logger.info(f"Selected best model: {best_model_name} with F1 score: {final_metrics['f1']:.4f}")
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Log the best overall model
    with mlflow.start_run(run_name="Best Model", nested=True) as best_run:
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_params({f"param_{k}": v for k, v in final_best_model.get_params().items()})
        
        for metric_name, metric_value in final_metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Save the best model to file
        model_path = os.path.join(model_dir, "best_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(final_best_model, f)
        logger.info(f"Best model saved to {model_path}")
        
        # Save the preprocessing components (scaler and label encoders)
        scaler_path = os.path.join(model_dir, "standard_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(data['scaler'], f)
        logger.info(f"StandardScaler saved to {scaler_path}")
        
        encoders_path = os.path.join(model_dir, "encoders.pkl")
        with open(encoders_path, "wb") as f:
            pickle.dump(data['encoders'], f)
        logger.info(f"Encoders saved to {encoders_path}")
        
        # Save preprocessing metadata for inference
        preprocessing_metadata = {
            "categorical_columns": data['categorical_cols'],
            "numerical_columns": data['numerical_cols']
        }
        metadata_path = os.path.join(model_dir, "preprocessing_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(preprocessing_metadata, f, indent=4)
        logger.info(f"Preprocessing metadata saved to {metadata_path}")
        
        # Log the model, scaler, and encoders as artifacts
        sign = infer_signature(data['X_test'], final_best_model.predict(data['X_test']))
        model_info = mlflow.sklearn.log_model(final_best_model, "Final_Best_Model", signature=sign)
        
        # Log artifacts
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(scaler_path)
        mlflow.log_artifact(encoders_path)
        mlflow.log_artifact(metadata_path)
        mlflow.log_artifact(__file__)
        
        # Create and log comparison table
        comparison_df = pd.DataFrame(best_metrics).T
        comparison_path = "model_comparison.csv"
        comparison_df.to_csv(comparison_path)
        mlflow.log_artifact(comparison_path)
        logger.info(f"Model comparison saved to {comparison_path}")
        
        # Log input data
        train_df = pd.concat([data['X_train_smote'], 
                             pd.DataFrame(data['y_train_smote'], columns=['readmitted'])], axis=1)
        test_df = pd.concat([data['X_test'], 
                            pd.DataFrame(data['y_test'], columns=['readmitted'])], axis=1)
        
        mlflow_train_df = mlflow.data.from_pandas(train_df)
        mlflow_test_df = mlflow.data.from_pandas(test_df)
        
        mlflow.log_input(mlflow_train_df, "train")
        mlflow.log_input(mlflow_test_df, "test")
        
        # Register the model in MLflow Model Registry
        model_name = "Readmission_Prediction_Model"
        
        # Register model to Model Registry
        model_details = mlflow.register_model(
            model_uri=f"runs:/{best_run.info.run_id}/Final_Best_Model",
            name=model_name
        )
        
        model_version = model_details.version
        logger.info(f"Model registered as '{model_name}' version {model_version}")
        
        # Add description and tags
        client.update_model_version(
            name=model_name,
            version=model_version,
            description=f"Best model: {best_model_name} with F1 score: {final_metrics['f1']:.4f}"
        )
        
        # Automatically transition to staging
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage="Staging"
        )
        
        # If the model exceeds performance threshold, promote to production
        f1_threshold = 0.65  # Adjust threshold based on your requirements
        if final_metrics['f1'] > f1_threshold:
            client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage="Production"
            )
            logger.info(f"Model version {model_version} promoted to Production automatically")
        else:
            logger.info(f"Model version {model_version} moved to Staging for review")
        
        # Save model registry information
        registry_info = {
            "model_name": model_name,
            "model_version": model_version,
            "best_model_type": best_model_name,
            "f1_score": final_metrics['f1']
        }
        
        registry_path = os.path.join(model_dir, "model_registry_info.json")
        with open(registry_path, "w") as f:
            json.dump(registry_info, f, indent=4)
        
        mlflow.log_artifact(registry_path)
        logger.info(f"Model registry info saved to {registry_path}")
        
        return final_metrics, model_version, best_model_name

def main():
    """Main function to run the training pipeline"""
    # Data and model paths
    FILE_PATH = "data/processed/processed_data.csv"
    REPO_PATH = "/home/rohith-ramanan/Desktop/DA5402/Patient_Readmission_Prediction"
    VERSION = "v20250428_224705"
    MODEL_DIR = "../../serving/backend/"
    CONFIG_PATH = "../config/model_params.yml"
    
    # Load model configurations
    config = load_config(CONFIG_PATH)
    model_configs = config['models']
    
    # Load data
    df = load_data(FILE_PATH, REPO_PATH, VERSION)
    logger.info(f"Loaded data shape: {df.shape}")
    
    # Preprocess data
    data = preprocess_data(df)
    
    # Train and evaluate models
    best_models, best_metrics, client, parent_run_id = train_and_evaluate_models(data, model_configs)
    
    # Select and register the best model
    final_metrics, model_version, best_model_name = select_and_register_best_model(
        best_models, best_metrics, client, data, MODEL_DIR)
    
    # Print final results
    logger.info("\n==== FINAL RESULTS ====")
    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"F1 Score: {final_metrics['f1']:.4f}")
    logger.info(f"Accuracy: {final_metrics['accuracy']:.4f}")
    logger.info(f"Precision: {final_metrics['precision']:.4f}")
    logger.info(f"Recall: {final_metrics['recall']:.4f}")
    logger.info(f"All models and results are tracked in MLflow with parent run ID: {parent_run_id}")

if __name__ == "__main__":
    # Setup logging
    logger.info("Starting model training pipeline")
    
    try:
        main()
        logger.info("Model training pipeline completed successfully")
    except Exception as e:
        logger.error(f"Error in model training pipeline: {e}", exc_info=True)
        raise