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