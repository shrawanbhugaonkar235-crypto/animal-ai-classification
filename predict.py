from project import load_model, predict_image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("image", help="Input image path")
args = parser.parse_args()

model = load_model()
predict_image(model, args.image)
