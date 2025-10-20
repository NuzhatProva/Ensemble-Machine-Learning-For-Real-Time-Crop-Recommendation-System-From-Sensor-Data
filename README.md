# Ensemble-Machine-Learning-For-Real-Time-Crop-Recommendation-System-From-Sensor-Data

- Generates 20-class 28×28 RGB images with simple, class-specific patterns.
- Trains a small `RandomForestClassifier` (scikit-learn).
- Exposes `/train`, `/predict`, `/labels`, `/health` endpoints via FastAPI.

## Endpoints
- `GET /health` → status
- `GET /labels` → current (or default) class labels
- `POST /train` → (re)generate synthetic dataset and train a fresh model
  - Query params: `n_classes` (2–20), `samples_per_class` (5–200), `image_size` (16–64), `test_size` (0.1–0.5)
- `POST /predict` → multipart upload of an image (`file`); returns predicted class + probabilities

## Local run
```bash
pip install -r requirements.txt
uvicorn app:app --port 8000 --host 0.0.0.0
# then call training once:
curl -X POST "http://localhost:8000/train?n_classes=20&samples_per_class=30"
```

## Render deployment
1. Create a **Web Service** on Render, connect this repo or upload as a private repo/zip.
2. **Build command** (auto):
   ```bash
   pip install -r requirements.txt
   ```
3. **Start command**:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port $PORT
   ```
4. After deploy, hit `/train` once to initialize the model:
   - `POST https://<your-service>.onrender.com/train?n_classes=20&samples_per_class=30`
5. Use `/predict` with an image.

> Tip: start small (e.g., `samples_per_class=15`) if you want faster training on cold boots.
