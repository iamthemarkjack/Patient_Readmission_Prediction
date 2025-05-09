# Patient Readmission Prediction - EP21B030 - NA21B050
## Submitted in partial fulfillment of the requirements for the course DA5402

An AI application designed to predict hospital readmission risk using patient demographic and clinical data. This system supports healthcare professionals in identifying high-risk patients and optimizing care pathways to reduce readmission rates.

## 🔍 Overview

This project integrates a complete MLOps pipeline using:
- **Apache Airflow** for orchestration (data loading & feature engineering)
- **DVC** for data versioning
- **MLflow** for model experiment tracking
- **FastAPI** as the model-serving backend
- **Streamlit** as the user-friendly frontend

## 📁 Project Structure

```
Patient_Readmission_Prediction/
├── airflow/                 # Orchestration with Apache Airflow
├── data/                    # Raw and processed datasets
├── models/                  # Model training & experiment configurations
├── notebooks/               # EDA and data analysis scripts
├── serving/                 # Backend (FastAPI), frontend (Streamlit)
├── tests/                   # Unit tests for feature modules
├── project_structure.txt    # File detailing the project directory
├── flowchart.drawio(.png)   # ML pipeline flowchart
├── Design_Document.pdf      # High/Low-level system design
├── User_Manual.pdf          # End-user instructions for app use
├── requirements.txt         # Python dependencies
└── README.md                # Project description and usage guide
```

## 🚀 Running the Application

1. **Clone the repository:**
   ```bash
   git clone https://github.com/iamthemarkjack/Patient_Readmission_Prediction
   cd Patient_Readmission_Prediction/serving
   ```

2. **Build & Deploy using Docker Compose:**
   ```bash
   docker compose up
   ```

This will start both the **backend (FastAPI)** and **frontend (Streamlit)** services.

## 🖥️ Features

- **Single Prediction**: Predict readmission risk for one patient
- **Batch Prediction**: Upload a CSV to predict risks for multiple patients
- **Versioned Models**: Every model and data version is tracked using MLflow and DVC
- **Interactive UI**: Built with Streamlit for ease of use by healthcare professionals

## 📚 Documentation

- **User Guide**: [User_Manual.pdf](./User_Manual.pdf)
- **System Design**: [Design_Document.pdf](./Design_Document.pdf)
