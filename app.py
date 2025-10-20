
import io
import os
import pickle
import numpy as np
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from typing import List, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from data import generate_dataset

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "model.pkl")
META_PATH = os.path.join(APP_DIR, "meta.pkl")

app = FastAPI(title="Rice Synthetic Classifier (Demo)",
              description="Tiny synthetic dataset + scikit-learn classifier for Render deployment",
              version="0.1.0")

class PredictResponse(BaseModel):
    predicted_class: str
    predicted_index: int
    probabilities: List[float]

class TrainResponse(BaseModel):
    n_classes: int
    samples_per_class: int
    image_size: int
    train_size: int
    val_size: int
    val_accuracy: float
    classes: List[str]

def _ensure_model():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
        raise RuntimeError("Model not trained yet. Call /train first.")

def _load_model():
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return clf, meta

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/labels")
def labels():
    try:
        _, meta = _load_model()
        return {"classes": meta["classes"]}
    except Exception:
        # default list when not trained yet
        n = 20
        classes = [
            "Subol_Lota", "Bashmoti", "Ganjiya", "Shampakatari", "Katarivog",
            "BR28", "BR29", "Paijam", "Bashful", "Lal_Aush",
            "Jirashail", "Gutisharna", "Red_Cargo", "Najirshail", "Katari_Polao",
            "Lal_Biroi", "Chinigura_Polao", "Amon", "Shorna5", "Lal_Binni"
        ][:n]
        return {"classes": classes}

@app.post("/train", response_model=TrainResponse)
def train(
    n_classes: int = Query(20, ge=2, le=20),
    samples_per_class: int = Query(20, ge=5, le=200),
    image_size: int = Query(28, ge=16, le=64),
    test_size: float = Query(0.25, ge=0.1, le=0.5),
    random_state: int = 42,
):
    X, y, class_names = generate_dataset(n_classes=n_classes, samples_per_class=samples_per_class, image_size=image_size)
    # Flatten images for a lightweight model
    Xf = X.reshape(len(X), -1).astype(np.float32) / 255.0

    Xtr, Xval, ytr, yval = train_test_split(Xf, y, test_size=test_size, stratify=y, random_state=random_state)

    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(Xtr, ytr)
    preds = clf.predict(Xval)
    acc = float(accuracy_score(yval, preds))

    # Persist
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    with open(META_PATH, "wb") as f:
        pickle.dump({"classes": class_names, "image_size": image_size}, f)

    return TrainResponse(
        n_classes=n_classes,
        samples_per_class=samples_per_class,
        image_size=image_size,
        train_size=len(Xtr),
        val_size=len(Xval),
        val_accuracy=acc,
        classes=class_names
    )

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    _ensure_model()
    clf, meta = _load_model()

    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB").resize((meta["image_size"], meta["image_size"]))
    arr = np.asarray(img).astype(np.float32).reshape(1, -1) / 255.0

    proba = clf.predict_proba(arr)[0]
    idx = int(np.argmax(proba))
    label = meta["classes"][idx]
    return PredictResponse(predicted_class=label, predicted_index=idx, probabilities=proba.tolist())
