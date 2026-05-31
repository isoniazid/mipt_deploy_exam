import glob
import os
import joblib
import pandas as pd
import time


from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles 
from fastapi.responses import FileResponse
from fastapi.responses import Response
from pydantic import BaseModel

import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST



MODEL_FILES = glob.glob("*.pkl")

if not MODEL_FILES:
    raise Exception("No model .pkl files found")

LATEST_MODEL = max(MODEL_FILES, key=os.path.getctime)

REQUEST_COUNT = Counter(
    "app_requests_total",
    "Total number of requests"
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Request latency"
)

CPU_USAGE = Gauge(
    "system_cpu_usage_percent",
    "CPU usage percent"
)

RAM_USAGE = Gauge(
    "system_ram_usage_percent",
    "RAM usage percent"
)

MODEL_PREDICTIONS = Counter(
    "model_predictions_total",
    "Total model predictions",
    ["class"]
)



print(f"Loading model: {LATEST_MODEL}")

pipeline = joblib.load(LATEST_MODEL)

app = FastAPI(
    title="Mushroom Toxicity API",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

class MushroomRequest(BaseModel):
    cap_texture: str
    spore_pattern: str
    stem_flexibility: str
    ring_thickness: str
    cap_shape: str
    cap_surface: str
    cap_color: str
    stalk_shape: str
    veil_type: str
    veil_color: str
    ring_number: str
    population: str
    habitat: str

@app.middleware("http")
async def metrics_middleware(request, call_next):
    start = time.time()

    response = await call_next(request)

    latency = time.time() - start

    REQUEST_COUNT.inc()
    REQUEST_LATENCY.observe(latency)

    return response


@app.get("/metrics")
def metrics():
    # системные метрики
    CPU_USAGE.set(psutil.cpu_percent())
    RAM_USAGE.set(psutil.virtual_memory().percent)
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/health")
def healthcheck():
    return {
        "status": "healthy",
        "model": LATEST_MODEL
    }

@app.get("/model")
def model_info():
    return {
        "loaded_model": LATEST_MODEL
    }

@app.get("/") 
def root(): 
    return FileResponse("static/index.html")

@app.post("/predict")
def predict(data: MushroomRequest):

    try:
        input_df = pd.DataFrame([{
            "cap_texture": data.cap_texture,
            "spore_pattern": data.spore_pattern,
            "stem_flexibility": data.stem_flexibility,
            "ring_thickness": data.ring_thickness,
            "cap_shape": data.cap_shape,
            "cap_surface": data.cap_surface,
            "cap_color": data.cap_color,
            "stalk_shape": data.stalk_shape,
            "veil_type": data.veil_type,
            "veil_color": data.veil_color,
            "ring_number": data.ring_number,
            "population": data.population,
            "habitat": data.habitat
        }])

        toxic_probability = float(pipeline.predict_proba(input_df)[0][1])
        is_toxic = toxic_probability >= 0.5

        label = "toxic" if is_toxic else "safe"

        MODEL_PREDICTIONS.labels(label).inc()

        return {
            "prediction": label,
            "confidence": round(toxic_probability, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))