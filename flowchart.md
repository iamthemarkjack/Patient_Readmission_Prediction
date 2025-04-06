```mermaid
flowchart TD
    subgraph Data Layer
        S1[Data Sources] --> DG[Data Gathering
                                Airflow DAGs]
        DG --> DVC[Data Version Control
                    DVC]
        DVC --> FS[Feature Store]
    end
    
    subgraph ML Pipeline
        FS --> FE[Feature Engineering]
        FE --> TR[Model Training]
        TR --> EV[Model Evaluation]
        EV --> REG[Model Registry
                    MLflow]
    end
    
    subgraph Deployment Layer
        REG --> DEPLOY[Model Deployment]
        DEPLOY --> PRED[Model Inference API]
    end
    
    subgraph Monitoring
        PRED --> PM[Performance Monitoring]
        PRED --> CD[Concept Drift Detection]
        PM --> PROM[Prometheus]
        CD --> PROM
        PROM --> AM[AlertManager]
        AM --> ALERT[Alert Notification]
    end
    
    subgraph Feedback Loop
        ALERT --> RETRAIN[Retraining Trigger]
        RETRAIN --> FE
    end
