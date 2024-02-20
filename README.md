# CVDL – Facial Expression Recognition

## Requirements

1. Install the needed requirements using pip:

```bash
pip install -r requirements.txt
```

2. Put the affectnet dataset into `./data/affectnet`, the fer2013 dataset into `./data/fer2013`, and the rafdb dataset
   into `./data/raf_db`. The given validation set should be put into `./data/test`. You can also change the paths
   in `.env` to your liking.

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

- `<webcam>` use webcam as input, default is `False`
- `cam-id` specify the webcam id, default is `0`
- `<input-video>` set a video file as input, required if not using webcam, will also generate an output video
- `<show-processing>` show the processing of the video, default is `True`
- `<explainability>` show the explanation of the prediction, default is `False`
- `<landmarks>` show remarkable landmarks, default is `False`
- `<info>` show information about the prediction, default is `True`
- `<codec>` specify the fourcc codec for the output video, default is `XVID`
- `<output-extension>` specify the output extension for the video, default is `avi`, depending on the codec

## Other utilities

### Clip faces from AffectNet

```bash
python main.py clip
```

Optional Arguments:

- `<output-dir>` directory to store the clipped faces, default is the path to the AffectNet dataset as specified
  in `.env`
- `<use-rafdb-format>` use the RAF-DB format for the output, default is `False`

### Create an ensemble of models

```bash
python main.py ensemble
```

Optional Arguments:

- `<data-path>` path to the data to be analyzed, default is the validation set (see `.env`)

### Initialize a hyperparameter sweep with wandb

```bash
python main.py sweep
```

Optional Arguments:

- `<count>` number of runs to be performed on the sweep
- `<sweep-id>` sweep id
- sweep config to be defined in `sweep.py`

### Prediction with activation values distribution
#### Get true value distributions

```bash
python main.py true-value-distributions <model-identifier>
```

Optional Arguments:

- `<data_path>` path to the data to be get the true value distributions from, default set to RAF-DB (see `.env`)
- `<output_path>` path for distributions and plots to be saved, default set to activation_values (see `.env`)
  
#### Analyze model performance with kl-divergence

```bash
kl-analyze <model-identifier>
```

Optional arguments:

- `<data_path>` path to the data to be get the true value distributions from, default set to validation set (see `.env`)
- `<output_path>` path for distributions and plots to be saved, default set to activation_values (see `.env`)
