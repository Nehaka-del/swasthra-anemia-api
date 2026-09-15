
import io
import json
import numpy as np
import tensorflow as tf

from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware


MODEL_PATH = "swasthra_anemia_model_current.keras"
IMG_SIZE = 224
THRESHOLD = 0.1063

model = tf.keras.models.load_model(MODEL_PATH)

app = FastAPI(
    title="SWASTHRA Anemia Screening API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_image(image_bytes):

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    return image


@app.get("/")
def root():

    return {
        "project": "SWASTHRA",
        "status": "online",
        "model": "MobileNetV2",
        "screening_only": True
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    metadata: str = Form("{}")
):

    image_bytes = await image.read()

    processed_image = preprocess_image(image_bytes)

    prediction = float(
        model.predict(processed_image, verbose=0)[0][0]
    )

    anemia_probability = prediction

    if anemia_probability >= THRESHOLD:
        visual_risk = "high"
    else:
        visual_risk = "low"

    try:
        metadata_json = json.loads(metadata)
    except Exception:
        metadata_json = {}

    return {
        "anemia_probability": round(anemia_probability, 4),
        "visual_risk": visual_risk,
        "threshold": THRESHOLD,
        "model": "SWASTHRA-MobileNetV2-v1",
        "screening_only": True,
        "metadata_received": metadata_json
    }
