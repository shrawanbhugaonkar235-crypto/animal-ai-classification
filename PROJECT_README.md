# AI-Based Classification and Localization of Animals, Sea Animals and Birds

This repository is the **actual Python AI project**, not just a website.

## What it does

Given an image, the project can:

- detect multiple objects
- identify the class supplied by the AI model
- group supported classes into Land Animal / Sea Animal / Bird
- calculate an individual prediction confidence
- generate bounding boxes
- save an annotated output image
- save a machine-readable JSON result
- run webcam inference
- train a custom YOLO model on a multi-species dataset

## Important accuracy rule

A prediction confidence such as 94.2% is **not** overall model accuracy.

Overall accuracy must be measured against a validation/test dataset. This project never generates fake accuracy values.

## Model strategy

### Quick testing
If `models/best.pt` exists, it is loaded.

Otherwise the program can load an Ultralytics pretrained model automatically.

### Full project requirement
To identify the complete requested collection of land animals, sea animals and birds, use a custom dataset and train/fine-tune a YOLO detector with those exact classes.

The repository includes `dataset/data.yaml.example` as a starting configuration.

## Run

Create environment:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r backend/requirements.txt
```

Run image detection:

```powershell
python predict.py "images/tiger.jpg"
```

Run full project:

```powershell
python project.py --image "images/tiger.jpg"
```

Run camera:

```powershell
python project.py --camera
```

Train:

```powershell
python train.py --data dataset/data.yaml --epochs 50
```

After training, use the resulting best.pt as:

```text
models/best.pt
```

## Output

Results are saved in:

```text
outputs/
  <image>_detected.jpg
  <image>_result.json
```

The JSON contains:

- object name
- major category
- confidence percentage
- confidence status
- x1/y1/x2/y2 bounding-box coordinates
- processing time

## Dataset structure

```text
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Each image needs a YOLO label file with:

```text
class_id center_x center_y width height
```

## Suggested intended categories

Land animals: dog, cat, lion, tiger, leopard, cheetah, elephant, horse, cow, buffalo, goat, sheep, deer, monkey, gorilla, bear, panda, wolf, fox, rabbit, zebra, giraffe, camel, kangaroo, hippopotamus, rhinoceros, pig, donkey, squirrel, rat, mouse.

Sea animals: shark, dolphin, whale, orca, seal, sea lion, walrus, octopus, squid, jellyfish, starfish, seahorse, crab, lobster, shrimp, turtle, sea turtle, stingray, eel, swordfish, tuna, salmon, clownfish, goldfish, pufferfish, angelfish, manta ray.

Birds: eagle, sparrow, crow, parrot, peacock, pigeon, owl, hawk, falcon, swan, duck, goose, flamingo, penguin, ostrich, emu, kingfisher, woodpecker, hummingbird, parakeet, pelican, seagull, crane, heron, rooster, hen, turkey.

Do not label a class as supported unless it exists in the trained model.
