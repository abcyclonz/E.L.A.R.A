FROM python:3.11-slim
WORKDIR /app
COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt
COPY . .
RUN mkdir -p tables
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
