FROM python:3.12-slim

WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY fixed_price_model.keras .
COPY hourly_rate_model.keras .
COPY country_encoder.joblib .

# Copy main app files
COPY main.py .
COPY index.html .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]