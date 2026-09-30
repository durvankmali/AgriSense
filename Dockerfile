FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.14.0 torchvision==0.29.0

RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY src ./src
COPY models ./models
COPY data/processed/agrisense_metadata_clean.csv ./data/processed/agrisense_metadata_clean.csv

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
