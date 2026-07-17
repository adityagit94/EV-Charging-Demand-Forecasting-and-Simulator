FROM python:3.12-slim
WORKDIR /app

# curl is needed for the container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "ev_forecast.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
