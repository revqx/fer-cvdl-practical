# CVDL – Facial Expression Recognition

## Usage

Install dependencies

```bash
pip install -r requirements.txt
```

Training using the config from `main.py`, possibly overriden by command line arguments

```bash
python main.py train
```

Inference using a trained model

```bash
python main.py inference <model-identifier> <input-path> <output-path>
```

- `<model-identifier>` specifies the model either by architecture or by W&B ID
- `<input-path>` root for all pictures to be tested
- `<output-path>` destination for `.csv` with results

Analysis of trained model

```bash
python main.py analyze <model-identifier> <input-path>
```

Video Prediction (make sure download the haar cascade from [here](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml) and put it into the `/cascades` folder)

```bash
python main.py video <model-identifier> <output-path>
```

optional arguments:
- `<webcam>` if using webcam
- `<input>` set a video file as input
- `<show-processing>` show the processing of the video
