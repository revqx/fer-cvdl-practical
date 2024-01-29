# CVDL – Facial Expression Recognition

## Requirements

1. Install the needed requirements using pip:

```bash
pip install -r requirements.txt
```

2. Put the affectnet dataset into `./data/affectnet`, the fer2013 dataset into `./data/fer2013`, and the rafdb dataset into `./data/raf_db`. The given validation set should be put into `./data/test`. You can also change the paths in `.env` to your liking.

## Train

```bash
python main.py train
```

Optional Arguments:

- `<offline>` if you want to train offline (e.g. without W&B), default is `False`

## Inference

```bash
python main.py inference <model-identifier> <input-path> <output-path>
```

Arguments:

- `<model-identifier>` specifies the model either by architecture or by W&B ID
- `<input-path>` root for all pictures to be tested
- `<output-path>` destination for `.csv` with results

## Analysis

```bash
python main.py analyze <model-identifier>
```

Optional Arguments:

- <data_path> path to the data to be analyzed, default is the validation set (see `.env`)

## Demo

```bash
python main.py demo <model-identifier> 
```

Optional arguments:
- `<record` record the demo, default is `False`
- `<webcam>` use webcam as input, default is `False`
- `cam-id` specify the webcam id, default is `0`
- `<input-file>` set a video file as input, required if not using webcam
- `<show-processing>` show the processing of the video, default is `True`
- `<explanation>` show the explanation of the prediction, default is `False`
- `<details>` show remarkable details of the face, default is `False`
- `<info>` show information about the prediction, default is `True`
- `<hog>` use HOG instead of haar cascade, not recommended for live demo, default is `False`


## Other utilities

### Clipping faces from AffectNet

```bash
python main.py clipped
```

Optional Arguments:

- `<output-dir>` directory to store the clipped faces, default is the path to the AffectNet dataset as specified in `.env`
- `<use-rafdb-format>` use the RAF-DB format for the output, default is `False`

### Creating an ensemble of models

```bash
python main.py ensemble
```

Optional Arguments:

- `<data-path>` path to the data to be analyzed, default is the validation set (see `.env`)

### Initializing a hyperparametersweep

```bash
python main.py initialize_sweep <user_name>
```

Optional Arguments:

- `<count>` number of runs to be performed on the sweep
- sweep config to be defined in `sweep.py` 

