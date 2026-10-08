# AI Animal Classification & Localization

Professional B.Tech CSE project web app for **animal, sea-animal and bird detection, classification and localization**.

## Stack
- HTML5, CSS3, Vanilla JavaScript
- Python + FastAPI
- Ultralytics YOLO
- Pillow / NumPy

## Features
- Image upload and drag/drop
- Real AI inference through the Python backend
- Multiple-object detection
- Real bounding boxes
- Class/category and prediction confidence
- Search and confidence/category filters
- Model health and supported-class view
- No database required

## Run locally
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r backend/requirements.txt
uvicorn backend.app:app --reload
```
Open http://127.0.0.1:8000

## Model
Place a YOLO/Ultralytics model at `models/best.pt` or set `MODEL_PATH` in `.env`.

A prediction confidence is **not** overall model accuracy. Overall accuracy must be measured on a validation/test dataset.

## Deployment
The repository includes `render.yaml` for Render. For full land/sea/bird species coverage, deploy a custom-trained model containing those classes.