import base64
import io
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

load_dotenv()
ROOT = Path(__file__).resolve().parent
MODEL_PATH = os.getenv("MODEL_PATH", str(ROOT / "models" / "best.pt"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
MAX_FILE_SIZE_MB = float(os.getenv("MAX_FILE_SIZE_MB", "4"))

CATEGORY_MAP = {
    "bird": "Bird",
    "dog": "Land Animal", "cat": "Land Animal", "lion": "Land Animal",
    "tiger": "Land Animal", "leopard": "Land Animal", "cheetah": "Land Animal",
    "elephant": "Land Animal", "horse": "Land Animal", "cow": "Land Animal",
    "buffalo": "Land Animal", "goat": "Land Animal", "sheep": "Land Animal",
    "deer": "Land Animal", "monkey": "Land Animal", "gorilla": "Land Animal",
    "bear": "Land Animal", "panda": "Land Animal", "wolf": "Land Animal",
    "fox": "Land Animal", "rabbit": "Land Animal", "zebra": "Land Animal",
    "giraffe": "Land Animal", "camel": "Land Animal", "kangaroo": "Land Animal",
    "hippopotamus": "Land Animal", "rhinoceros": "Land Animal", "pig": "Land Animal",
    "donkey": "Land Animal", "squirrel": "Land Animal", "rat": "Land Animal",
    "mouse": "Land Animal",
    "shark": "Sea Animal", "dolphin": "Sea Animal", "whale": "Sea Animal",
    "orca": "Sea Animal", "seal": "Sea Animal", "sea lion": "Sea Animal",
    "walrus": "Sea Animal", "octopus": "Sea Animal", "squid": "Sea Animal",
    "jellyfish": "Sea Animal", "starfish": "Sea Animal", "seahorse": "Sea Animal",
    "crab": "Sea Animal", "lobster": "Sea Animal", "shrimp": "Sea Animal",
    "turtle": "Sea Animal", "sea turtle": "Sea Animal", "stingray": "Sea Animal",
    "eel": "Sea Animal", "swordfish": "Sea Animal", "tuna": "Sea Animal",
    "salmon": "Sea Animal", "clownfish": "Sea Animal", "goldfish": "Sea Animal",
    "pufferfish": "Sea Animal", "angelfish": "Sea Animal", "manta ray": "Sea Animal",
}

INTENDED = [
    ("Land Animals", ["Dog","Cat","Lion","Tiger","Leopard","Cheetah","Elephant","Horse","Cow","Buffalo","Goat","Sheep","Deer","Monkey","Gorilla","Bear","Panda","Wolf","Fox","Rabbit","Zebra","Giraffe","Camel","Kangaroo","Hippopotamus","Rhinoceros","Pig","Donkey","Squirrel","Rat","Mouse"]),
    ("Sea Animals", ["Shark","Dolphin","Whale","Orca","Seal","Sea Lion","Walrus","Octopus","Squid","Jellyfish","Starfish","Seahorse","Crab","Lobster","Shrimp","Turtle","Sea Turtle","Stingray","Eel","Swordfish","Tuna","Salmon","Clownfish","Goldfish","Pufferfish","Angelfish","Manta Ray"]),
    ("Birds", ["Eagle","Sparrow","Crow","Parrot","Peacock","Pigeon","Owl","Hawk","Falcon","Swan","Duck","Goose","Flamingo","Penguin","Ostrich","Emu","Kingfisher","Woodpecker","Hummingbird","Parakeet","Pelican","Seagull","Crane","Heron","Rooster","Hen","Turkey"]),
]

model = None
model_error = None
try:
    from ultralytics import YOLO
    if Path(MODEL_PATH).exists():
        model = YOLO(MODEL_PATH)
    else:
        model_error = f"Model file not found: {MODEL_PATH}"
except Exception as exc:
    model_error = str(exc)

@app.get("/")
def root():
    return {
        "name": "Animal Vision AI",
        "status": "running",
        "message": "Python AI API is online. Use /api/health, /api/classes and POST /api/predict.",
        "api_docs": "/docs"
    }

app = FastAPI(
    title="Animal Vision AI",
    version="1.0.0",
    description="Animal, sea-animal and bird classification/localization."
)

origins = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def confidence_status(c: float) -> str:
    if c >= 0.90: return "High Confidence"
    if c >= 0.70: return "Good Confidence"
    if c >= 0.50: return "Moderate Confidence"
    return "Low Confidence"

def get_model_names():
    if model is None:
        return []
    try:
        return [str(v) for v in model.names.values()]
    except Exception:
        return []

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_name": Path(MODEL_PATH).name,
        "model_error": model_error,
    }

@app.get("/api/classes")
def classes():
    names = get_model_names()
    supported = {n.strip().lower() for n in names}
    payload = []
    for group, values in INTENDED:
        for value in values:
            payload.append({
                "name": value,
                "category": group,
                "supported": value.strip().lower() in supported
            })
    return {
        "model_loaded": model is not None,
        "model_name": Path(MODEL_PATH).name,
        "actual_model_classes": names,
        "classes": payload
    }

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No image was selected.")
    ext = Path(file.filename).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(400, "Use JPG, JPEG, PNG or WEBP.")

    data = await file.read()
    if not data:
        raise HTTPException(400, "The image is empty.")
    if len(data) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"Image exceeds the {MAX_FILE_SIZE_MB:g} MB server limit.")

    try:
        check = Image.open(io.BytesIO(data))
        check.verify()
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, "The uploaded file is not a valid image.")

    if model is None:
        raise HTTPException(503, "AI model is not loaded. Add trained weights and configure MODEL_PATH.")

    request_id = uuid.uuid4().hex
    start = time.perf_counter()

    # The model accepts the in-memory image; no persistent upload storage is required.
    try:
        results = model.predict(source=image, conf=CONFIDENCE_THRESHOLD, verbose=False)
        result = results[0]
        detections = []
        names = result.names

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                label = str(names.get(cls_id, cls_id))
                category = CATEGORY_MAP.get(label.lower().strip(), "Other")
                detections.append({
                    "name": label,
                    "category": category,
                    "confidence": round(conf, 6),
                    "confidence_percent": round(conf * 100, 2),
                    "status": confidence_status(conf),
                    "bbox": {
                        "x1": round(xyxy[0], 2), "y1": round(xyxy[1], 2),
                        "x2": round(xyxy[2], 2), "y2": round(xyxy[3], 2)
                    }
                })

        annotated = result.plot()
        annotated_image = Image.fromarray(annotated[..., ::-1])
        buf = io.BytesIO()
        annotated_image.save(buf, format="JPEG", quality=90, optimize=True)
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:
        raise HTTPException(500, f"Prediction failed: {exc}")

    counts = {"Land Animal": 0, "Sea Animal": 0, "Bird": 0}
    for d in detections:
        counts[d["category"]] = counts.get(d["category"], 0) + 1

    avg = round(
        sum(d["confidence_percent"] for d in detections) / len(detections), 2
    ) if detections else 0

    return {
        "success": True,
        "request_id": request_id,
        "total_detections": len(detections),
        "summary": {
            "land_animals": counts["Land Animal"],
            "sea_animals": counts["Sea Animal"],
            "birds": counts["Bird"],
            "average_confidence": avg,
        },
        "detections": detections,
        "processing_time_ms": round((time.perf_counter() - start) * 1000, 2),
        "annotated_image": f"data:image/jpeg;base64,{image_b64}",
        "model_name": Path(MODEL_PATH).name,
        "note": "Confidence is per-prediction confidence, not overall model accuracy."
    }
