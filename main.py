from datetime import datetime

import typer
from dotenv import load_dotenv
import os
import wandb

from analyze import accuracies, confusion_matrix
from train import train_model
from inference import apply_model

load_dotenv()
app = typer.Typer()

DEFAULT_TRAIN_CONFIG = {
    "model_name": "LeNet",
    # Options: LeNet
    "model_description": "",
    "train_data": "AffectNet",
    "preprocessing": "StandardizeRGB()",  # everything done on the 64x64 tensors
    # Options: StandardizeGray(), StandardizeRGB()
    "black_and_white": False,  # switches between 1 and 3 channels
    "validation_split": 0.2,
    "learning_rate": 0.001,
    "sampler": "uniform",  # Options: uniform, None
    "epochs": 7,
    "batch_size": 64,
    "loss_function": "CrossEntropyLoss",
    "optimizer": "Adam",
    "device": "cpu",
}


@app.command()
def train(offline: bool = False):
    config = DEFAULT_TRAIN_CONFIG
    # disable wandb if offline
    os.environ['WANDB_MODE'] = 'offline' if offline else 'online'
    wandb.init(project="cvdl", config=config)
    model = train_model(config)
    print(model)


@app.command()
def inference(model_name: str, data_path: str, output_path: str):
    model_id, results = apply_model(model_name, data_path)
    # create name from model_name and timestamp and input files
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    folder_name = data_path.split('/')[-1]
    output_file_name = f"{folder_name}-{timestamp}-{model_id}.csv"
    print(f"Writing results to {output_file_name}.")
    results.to_csv(os.path.join(output_path, output_file_name), index=False)


@app.command()
def analyze(model_name: str, data_path: str):
    model_id, results = apply_model(model_name, data_path)
    top_n = accuracies(results, best=3)
    conf_matrix = confusion_matrix(results)
    print(conf_matrix)
    print(top_n)


if __name__ == "__main__":
    app()