# AI-Based Classification and Localization of Animals, Sea Animals and Birds Using Python

This repository contains the **actual Python computer-vision project**, not a website.

## Project objective

The system detects animals in an image or camera feed, identifies the class supplied by the trained model, localizes each object with a bounding box, and reports the confidence of each individual prediction.

The expanded project scope is:

- Land / general animals
- Sea / aquatic animals
- Birds

## Important accuracy rule

A prediction confidence such as `94.72%` is **not** the overall model accuracy.

Overall model accuracy must be calculated on a validation/test dataset. This project never creates random or fabricated accuracy numbers.

## Technology

- Python
- YOLO / Ultralytics
- OpenCV
- NumPy
- Pillow
- Matplotlib

## Project structure

```text
animal-ai-classification/
├── project.py
├── predict.py
├── train.py
├── evaluate.py
├── requirements.txt
├── .env.example
├── .python-version
├── models/
│   ├── best.pt              # your trained model
│   └── README.md
├── dataset/
│   └── data.yaml.example
├── outputs/
│   └── .gitkeep
└── README.md
```

## Installation

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Model

Set the model path through `.env`:

```text
MODEL_PATH=models/best.pt
CONFIDENCE_THRESHOLD=0.25
```

If `models/best.pt` exists, it is loaded.

For testing without a custom model, the project can use an Ultralytics pretrained model. However, a pretrained model can only predict the classes it was trained on.

## Image prediction

```powershell
python predict.py "images/tiger.jpg"
```

or:

```powershell
python project.py --image "images/tiger.jpg"
```

The program produces:

- annotated image with real bounding boxes
- JSON result containing class name, category, confidence, and coordinates
- processing time

## Camera detection

```powershell
python project.py --camera
```

## Custom training

For the complete land + sea + bird scope, create a YOLO dataset using the required exact classes.

Dataset layout:

```text
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Then create `dataset/data.yaml` based on `dataset/data.yaml.example`.

Train:

```powershell
python train.py --data dataset/data.yaml --epochs 50
```

After training, copy the best weights to:

```text
models/best.pt
```

## Evaluation

Run the validation metrics with:

```powershell
python evaluate.py --data dataset/data.yaml --model models/best.pt
```

This reports validation metrics from the trained model. Use those measured metrics in the academic report instead of a prediction confidence.

## Intended classes

### Land animals
Dog, Cat, Lion, Tiger, Leopard, Cheetah, Elephant, Horse, Cow, Buffalo, Goat, Sheep, Deer, Monkey, Gorilla, Bear, Panda, Wolf, Fox, Rabbit, Zebra, Giraffe, Camel, Kangaroo, Hippopotamus, Rhinoceros, Pig, Donkey, Squirrel, Rat, Mouse.

### Sea animals
Shark, Dolphin, Whale, Orca, Seal, Sea Lion, Walrus, Octopus, Squid, Jellyfish, Starfish, Seahorse, Crab, Lobster, Shrimp, Turtle, Sea Turtle, Stingray, Eel, Swordfish, Tuna, Salmon, Clownfish, Goldfish, Pufferfish, Angelfish, Manta Ray.

### Birds
Eagle, Sparrow, Crow, Parrot, Peacock, Pigeon, Owl, Hawk, Falcon, Swan, Duck, Goose, Flamingo, Penguin, Ostrich, Emu, Kingfisher, Woodpecker, Hummingbird, Parakeet, Pelican, Seagull, Crane, Heron, Rooster, Hen, Turkey.

A class is supported only when it exists in the actual trained model.

## Academic/project relevance

The source project focuses on animal classification, localization, bounding boxes, confidence estimation, multiple-animal detection, and Python/YOLO-based computer vision. This implementation keeps that core concept and extends the intended category scope to land animals, sea animals and birds.

## No fake data

Do not add:
- fake detection results
- random confidence values
- static bounding boxes
- fabricated validation accuracy

All predictions and localization coordinates must come from the loaded AI model.
