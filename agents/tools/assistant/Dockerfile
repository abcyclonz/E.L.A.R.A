FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py .
RUN mkdir -p /data
VOLUME ["/data"]
EXPOSE 8011
CMD ["python", "server.py"]
