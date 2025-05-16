# Vietnamese Food Recognition System

this project using CNN (convolutional neutral network) and YOLO (You only look once) to detect foods on the tray.

## Features

- Real-time recognition
- Flexible UI made by PyQt5
- Using a Webcam to detect or browse image manually

## Structures

```
.
├── src/                    # Source code
│   ├── data/              # Data handling utilities
│   ├── models/            # Model 
│   ├── training/          # Training scripts
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── notebooks/             # notebooks
├── tests/                 # testing
├── requirements.txt       # Dependencies
├── setup.py              # Package installation
└── main.py               # Entry point
```

## Installation

1. CLone this repo by using command:

```bash
git clone https://github.com/Ngducok/Food-Classfication.git

cd Food-Classfication
```

2. Usage 

```bash
python main.py train --config configs/train_config.yaml
```

### GUI

```bash
python run_gui.py
```

The dataset should be organized in the following structure:
```
Dataset_fix/
├── images/
│   ├── train/
│   │   ├── class1/
│   │   ├── class2/
│   │   └── ...
│   └── val/
│       ├── class1/
│       ├── class2/
│       └── ...
```

## Licence
This project is licensed under the Apache2.0 License - see the LICENSE file for details.