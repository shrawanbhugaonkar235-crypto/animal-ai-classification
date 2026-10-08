from project import train
import argparse

parser = argparse.ArgumentParser(description="Train a custom multi-species YOLO detector")
parser.add_argument("--data", required=True, help="Path to YOLO dataset YAML")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--model", default="yolo11n.pt")
args = parser.parse_args()

train(args.data, args.epochs, args.model)
