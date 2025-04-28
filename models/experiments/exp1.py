# Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC
import pickle
import mlflow
from mlflow.models.signature import infer_signature

# Load the data
data_path = '../data/processed/data_featured.csv'
df = pd.read_csv(data_path)

# Define the target and features
X = df.drop(columns=['readmitted'])  # Features
y = df['readmitted']  # Target

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Create label encoders for categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTENC to balance the data (for categorical + numerical data)
categorical_indices = [X.columns.get_loc(col) for col in categorical_cols]
smote = SMOTENC(categorical_features=categorical_indices, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Define the Random Forest Classifier model and the parameter distribution for hyperparameter tuning
rf = RandomForestClassifier(random_state=42)
param_dist = {
    'n_estimators': [100, 200, 300, 500, 1000],  # Different values of n_estimators to try
    'max_depth': [None, 4, 5, 6, 10],  # Different max_depth values to explore
}

# Perform RandomizedSearchCV to find the best hyperparameters for the Random Forest model
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=50, cv=5, n_jobs=-1, verbose=2, random_state=42)

# Start a new MLflow run to log the Random Forest tuning process
with mlflow.start_run(run_name="Random Forest Tuning") as parent_run:

    # Fit the RandomizedSearchCV object on the SMOTE-resampled training data
    random_search.fit(X_train_smote, y_train_smote)

    # Log the parameters and mean test scores for each combination tried
    for i in range(len(random_search.cv_results_['params'])):
        with mlflow.start_run(run_name=f"Combination{i+1}", nested=True) as child_run:
            mlflow.log_params(random_search.cv_results_['params'][i])  # Log the parameters
            mlflow.log_metric("mean_test_score", random_search.cv_results_['mean_test_score'][i])  # Log the mean test score

    # Print the best hyperparameters found by RandomizedSearchCV
    print("Best parameters found: ", random_search.best_params_)

    # Log the best parameters in MLflow
    mlflow.log_params(random_search.best_params_)

    # Train the model using the best parameters identified by RandomizedSearchCV
    best_rf = random_search.best_estimator_
    
    # No need to fit again as RandomizedSearchCV already fit the model
    # but if you want to train on the SMOTE data specifically:
    best_rf.fit(X_train_smote, y_train_smote)

    # Save the trained model to a file for later use
    pickle.dump(best_rf, open("model.pkl", "wb"))

    # Make predictions on the test set using the best model
    y_pred = best_rf.predict(X_test)

    # Calculate and print performance metrics: accuracy, precision, recall, and F1-score
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log performance metrics into MLflow for tracking
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1 score", f1)

    # Create DataFrames for logging as inputs
    train_df = pd.concat([X_train_smote, pd.DataFrame(y_train_smote, columns=['readmitted'])], axis=1)
    test_df = pd.concat([X_test, pd.DataFrame(y_test, columns=['readmitted'])], axis=1)
    
    # Log the training and testing data as inputs in MLflow
    mlflow_train_df = mlflow.data.from_pandas(train_df)
    mlflow_test_df = mlflow.data.from_pandas(test_df)
    
    mlflow.log_input(mlflow_train_df, "train")  # Log training data
    mlflow.log_input(mlflow_test_df, "test")  # Log test data

    # Log the current script file as an artifact in MLflow
    mlflow.log_artifact(__file__)

    # Infer the model signature using the test features and predictions
    sign = infer_signature(X_test, best_rf.predict(X_test))
    
    # Log the trained model in MLflow with its signature
    mlflow.sklearn.log_model(best_rf, "Best Model", signature=sign)

    # Print the calculated performance metrics to the console for review
    print("Accuracy: ", acc)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-score: ", f1)