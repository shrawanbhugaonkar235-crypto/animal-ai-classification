import io
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
UPLOADS = ROOT / "uploads"
OUTPUTS = ROOT / "outputs"
MODEL_PATH = os.getenv("MODEL_PATH", str(ROOT / "models" / "best.pt"))
THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
MAX_MB = float(os.getenv("MAX_FILE_SIZE_MB", "10"))

CATEGORY = {
    "bird": "Bird", "dog": "Land Animal", "cat": "Land Animal",
    "horse": "Land Animal", "cow": "Land Animal", "sheep": "Land Animal",
    "elephant": "Land Animal", "bear": "Land Animal", "zebra": "Land Animal",
    "giraffe": "Land Animal", "lion": "Land Animal", "tiger": "Land Animal",
    "shark": "Sea Animal", "dolphin": "Sea Animal", "whale": "Sea Animal",
    "octopus": "Sea Animal", "squid": "Sea Animal", "jellyfish": "Sea Animal",
    "starfish": "Sea Animal", "seahorse": "Sea Animal", "crab": "Sea Animal",
    "lobster": "Sea Animal", "turtle": "Sea Animal", "seal": "Sea Animal",
    "orca": "Sea Animal", "stingray": "Sea Animal", "manta ray": "Sea Animal"
}

MODEL_ERROR = None
model = None
try:
    from ultralytics import YOLO
    if Path(MODEL_PATH).exists():
        model = YOLO(MODEL_PATH)
    else:
        MODEL_ERROR = f"Model file not found: {MODEL_PATH}"
except Exception as exc:
    MODEL_ERROR = str(exc)

app = FastAPI(title="AI Animal Classification & Localization", version="1.0.0")
origins = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

def status(c):
    return "High Confidence" if c >= .9 else "Good Confidence" if c >= .7 else "Moderate Confidence" if c >= .5 else "Low Confidence"

@app.get("/")
def home():
    return FileResponse(FRONTEND / "index.html")

@app.get("/api/health")
def health():
    return {"status":"ok", "model_loaded": model is not None, "model": Path(MODEL_PATH).name, "error": MODEL_ERROR}

@app.get("/api/classes")
def classes():
    names = []
    if model is not None:
        try: names = [str(v) for v in model.names.values()]
        except Exception: pass
    return {"model_loaded": model is not None, "actual_model_classes": names}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".jpg",".jpeg",".png",".webp"}:
        raise HTTPException(400, "Use JPG, JPEG, PNG or WEBP.")
    data = await file.read()
    if not data or len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(400, f"Invalid or oversized image. Maximum {MAX_MB:g} MB.")
    try:
        check = Image.open(io.BytesIO(data))
        check.verify()
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, "The uploaded file is not a valid image.")
    if model is None:
        raise HTTPException(503, "AI model is not loaded. Configure MODEL_PATH and add the model weights.")
    rid = uuid.uuid4().hex
    src = UPLOADS / f"{rid}{ext}"
    out = OUTPUTS / f"{rid}.jpg"
    src.write_bytes(data)
    t0 = time.perf_counter()
    try:
        result = model.predict(source=str(src), conf=THRESHOLD, verbose=False)[0]
        detections = []
        names = result.names
        for b in result.boxes or []:
            cls = int(b.cls.item())
            conf = float(b.conf.item())
            xy = [float(v) for v in b.xyxy[0].tolist()]
            label = str(names.get(cls, cls))
            detections.append({
                "class_name": label, "category": CATEGORY.get(label.lower(), "Other"),
                "confidence_percent": round(conf*100,2), "status": status(conf),
                "bbox": {"x1":round(xy[0],2),"y1":round(xy[1],2),"x2":round(xy[2],2),"y2":round(xy[3],2)}
            })
        annotated = result.plot()
        Image.fromarray(annotated[..., ::-1]).save(out, quality=92)
    finally:
        src.unlink(missing_ok=True)
    counts = {"Land Animal":0,"Sea Animal":0,"Bird":0}
    for d in detections: counts[d["category"]] = counts.get(d["category"],0)+1
    avg = round(sum(d["confidence_percent"] for d in detections)/len(detections),2) if detections else 0
    return {
        "success":True, "total_detections":len(detections), "detections":detections,
        "summary":{"land_animals":counts["Land Animal"],"sea_animals":counts["Sea Animal"],"birds":counts["Bird"],"average_confidence":avg},
        "processing_time_ms":round((time.perf_counter()-t0)*1000,2),
        "annotated_image_url":f"/api/output/{out.name}", "model_name":Path(MODEL_PATH).name
    }

@app.get("/api/output/{name}")
def output(name:str):
    p = OUTPUTS / Path(name).name
    if not p.exists(): raise HTTPException(404, "Result not found.")
    return FileResponse(p, media_type="image/jpeg")