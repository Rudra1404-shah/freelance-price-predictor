import os
os.environ['USE_TF'] = '0'  # force sentence-transformers to use PyTorch, not TensorFlow

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from pathlib import Path

app = FastAPI(
    title="Freelance Price Predictor",
    description="Predicts avg price for freelance jobs using neural networks + text embeddings"
)

BASE_DIR = Path(__file__).resolve().parent


fixed_model   = tf.keras.models.load_model(BASE_DIR / 'fixed_price_model.keras')
hourly_model  = tf.keras.models.load_model(BASE_DIR / 'hourly_rate_model.keras')
country_enc   = joblib.load(BASE_DIR / 'country_encoder.joblib')
embed_model   = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


class JobInput(BaseModel):
    job_title: str
    job_description: str
    tags: str
    client_country: str
    rate_type: str  # 'fixed' or 'hourly'


@app.get("/")
def root():
    return {"message": "Freelance Price Predictor API is running"}


@app.post("/predict")
def predict(job: JobInput):
    if job.rate_type not in ("fixed", "hourly"):
        raise HTTPException(status_code=400, detail="rate_type must be 'fixed' or 'hourly'")

    # Combine text exactly as done during training
    text = f"{job.job_title} | {job.job_description} | {job.tags}"

    # Generate embedding
    embedding = embed_model.encode([text])  # shape (1, 768)

    # Encode country — use 0 for unseen countries
    try:
        country_code = country_enc.transform([job.client_country])[0]
    except ValueError:
        country_code = 0

    X = {
        'text_embeddings': embedding,
        'country': np.array([[country_code]])
    }

    model = fixed_model if job.rate_type == 'fixed' else hourly_model

    log_pred = model.predict(X, verbose=0)
    price = float(np.expm1(log_pred[0][0]))

    return {
        "predicted_avg_price_usd": round(price, 2),
        "rate_type": job.rate_type,
        "client_country": job.client_country
    }


@app.get("/countries")
def get_countries():
    return {"countries": list(country_enc.classes_)}
