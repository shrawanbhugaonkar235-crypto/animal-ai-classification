import argparse
from pathlib import Path
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model on a validation dataset.")
parser.add_argument("--data", required=True, help="Path to dataset/data.yaml")
parser.add_argument("--model", required=True, help="Path to trained model weights, e.g. models/best.pt")
args = parser.parse_args()

model = YOLO(str(Path(args.model)))
metrics = model.val(data=args.data, verbose=True)

print("\nValidation metrics")
print(f"mAP50:     {float(metrics.box.map50):.4f}")
print(f"mAP50-95:  {float(metrics.box.map):.4f}")
try:
    print(f"Precision: {float(metrics.box.mp):.4f}")
    print(f"Recall:    {float(metrics.box.mr):.4f}")
except Exception:
    pass

print("\nThese are validation metrics, not individual prediction confidence scores.")
