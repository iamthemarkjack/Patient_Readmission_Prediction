FROM python:3.9-slim

WORKDIR /app

COPY serving/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serving/backend/. .

# Copy the utility file into the image
COPY airflow/dags/utils/add_feature.py ./add_feature.py

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]