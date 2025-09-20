FROM python:3.13-slim

WORKDIR /app

COPY api/ .

COPY models/model.pkl /app/models/model.pkl
COPY models/scaler.pkl /app/models/scaler.pkl

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8501"]