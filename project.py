"""
AI Animal Classification & Localization Project
Run:
    python project.py --image path/to/image.jpg
    python project.py --camera
    python project.py --train --data dataset/data.yaml --epochs 50
"""

import argparse
import json
import os
import time
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = os.getenv("MODEL_PATH", str(ROOT / "models" / "best.pt"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "yolo11n.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

CATEGORY_MAP = {
    "bird": "Bird",
    "dog": "Land Animal", "cat": "Land Animal", "horse": "Land Animal",
    "cow": "Land Animal", "sheep": "Land Animal", "elephant": "Land Animal",
    "bear": "Land Animal", "zebra": "Land Animal", "giraffe": "Land Animal",
    "lion": "Land Animal", "tiger": "Land Animal", "leopard": "Land Animal",
    "cheetah": "Land Animal", "wolf": "Land Animal", "fox": "Land Animal",
    "deer": "Land Animal", "monkey": "Land Animal", "gorilla": "Land Animal",
    "panda": "Land Animal", "rabbit": "Land Animal", "camel": "Land Animal",
    "kangaroo": "Land Animal", "hippopotamus": "Land Animal",
    "rhinoceros": "Land Animal", "pig": "Land Animal", "donkey": "Land Animal",
    "squirrel": "Land Animal", "rat": "Land Animal", "mouse": "Land Animal",
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

def category(label: str) -> str:
    return CATEGORY_MAP.get(label.lower().strip(), "Other")

def status(conf: float) -> str:
    if conf >= .90: return "High Confidence"
    if conf >= .70: return "Good Confidence"
    if conf >= .50: return "Moderate Confidence"
    return "Low Confidence"

def load_model():
    custom = Path(MODEL_PATH)
    if custom.exists():
        print(f"Loading custom model: {custom}")
        return YOLO(str(custom))
    print(f"Custom model not found at {custom}.")
    print(f"Using Ultralytics pretrained model: {DEFAULT_MODEL}")
    print("Important: pretrained models only support their trained classes.")
    return YOLO(DEFAULT_MODEL)

def predict_image(model, image_path: str):
    start = time.perf_counter()
    results = model.predict(source=image_path, conf=CONFIDENCE_THRESHOLD, verbose=False)
    result = results[0]

    detections = []
    names = result.names

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            name = str(names.get(cls_id, cls_id))

            detections.append({
                "name": name,
                "category": category(name),
                "confidence_percent": round(conf * 100, 2),
                "status": status(conf),
                "bounding_box": {
                    "x1": round(xyxy[0], 2),
                    "y1": round(xyxy[1], 2),
                    "x2": round(xyxy[2], 2),
                    "y2": round(xyxy[3], 2),
                }
            })

    output = OUTPUT_DIR / f"{Path(image_path).stem}_detected.jpg"
    result.save(filename=str(output))

    report = {
        "input": str(image_path),
        "output": str(output),
        "total_objects": len(detections),
        "processing_time_ms": round((time.perf_counter() - start) * 1000, 2),
        "detections": detections,
        "note": "Confidence is per-prediction confidence, not overall model accuracy."
    }

    json_path = OUTPUT_DIR / f"{Path(image_path).stem}_result.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Annotated image: {output}")
    print(f"Result JSON: {json_path}")

def train(data_yaml: str, epochs: int, model_name: str):
    model = YOLO(model_name)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        project=str(ROOT / "runs"),
        name="animal_classifier"
    )
    print(results)
    print("After training, copy the best weights to models/best.pt")

def camera(model):
    # Ultralytics uses OpenCV/webcam internally; 0 is the default camera.
    results = model.predict(source=0, conf=CONFIDENCE_THRESHOLD, show=True, stream=True)
    for _ in results:
        pass

def main():
    parser = argparse.ArgumentParser(description="Animal classification and localization using Python")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to input image")
    group.add_argument("--camera", action="store_true", help="Run webcam inference")
    group.add_argument("--train", action="store_true", help="Train a custom model")
    parser.add_argument("--data", help="YOLO dataset YAML for custom training")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if args.train:
        if not args.data:
            parser.error("--train requires --data dataset/data.yaml")
        train(args.data, args.epochs, args.model)
        return

    model = load_model()

    if args.image:
        if not Path(args.image).exists():
            raise FileNotFoundError(f"Image not found: {args.image}")
        predict_image(model, args.image)
    else:
        camera(model)

if __name__ == "__main__":
    main()
