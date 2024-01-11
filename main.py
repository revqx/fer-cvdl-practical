from datetime import datetime

import typer
from dotenv import load_dotenv
import os
import wandb

from analyze import accuracies, confusion_matrix, analyze_run_and_upload
from train import train_model
from inference import apply_model
from video_prediction import make_video_prediction
from clip_affect_net import clip_affect_net_faces

load_dotenv()
app = typer.Typer()

# Default config for training should not be altered by the user
DEFAULT_TRAIN_CONFIG = {
    "model_name": "LeNet",
    # Options: LeNet, ResNet18
    "model_description": "",
    "train_data": "AffectNet",  # Options: AffectNet, RAF_DB
    "preprocessing": "StandardizeRGB()",  # everything done on the 64x64 tensors
    # Options: StandardizeGray(), StandardizeRGB()
    "black_and_white": False,  # switches between 1 and 3 channels
    "validation_split": 0.2,
    "learning_rate": 0.001,
    "sampler": "uniform",  # Options: uniform, None
    "patience": 3,
    "epochs": 7,
    "batch_size": 64,
    "loss_function": "CrossEntropyLoss",
    "optimizer": "Adam",
    "device": "cpu",
}

# If you want to use a custom config, change this one as you like
CUSTOM_TRAIN_CONFIG = {
    "model_name": "LeNet",
    # Options: LeNet, ResNet18
    "model_description": "",
    "train_data": "RAF-DB",
    # Options: AffectNet, RAF-DB
    "preprocessing": "",  # everything done on the 64x64 tensors
    # Options: StandardizeGray(), StandardizeRGB()
    "epochs": 10,
    "batch_size": 32,
}


@app.command()
def train(offline: bool = False):
    # merge default and custom config
    config = DEFAULT_TRAIN_CONFIG | CUSTOM_TRAIN_CONFIG
    # check if valiadation path is valid
    if not os.path.exists(os.getenv("DATASET_VALIDATION_PATH")):
        raise FileNotFoundError(f"Directory {os.getenv('DATASET_VALIDATION_PATH')} not found. "
                                f"Could not load validation dataset.")

    # disable wandb if offline
    os.environ['WANDB_MODE'] = 'offline' if offline else 'online'
    wandb.init(project="cvdl", config=config)
    _ = train_model(config)
    # test model on validation data
    analyze_run_and_upload(config["model_name"])


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
def analyze(model_name: str, data_path: str = os.getenv("DATASET_VALIDATION_PATH")):
    model_id, results = apply_model(model_name, data_path)
    top_n = accuracies(results, best=3)
    conf_matrix = confusion_matrix(results)
    print(conf_matrix)
    print(top_n)


@app.command()
def video(model_name: str, output_path: str, webcam: bool = False, input_: str = "", show_processing: bool = True):
    if not webcam and not input_:
        raise typer.BadParameter("Please specify a video input when not using the camera.")

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_file = os.path.join(output_path, f"{model_name}-{timestamp}.avi")
    make_video_prediction(model_name, webcam, input_, output_file, show_processing)


@app.command()
def clipped(output_dir: str = "data/clipped_affect_net"):
    input_path = os.getenv('DATASET_AFFECT_NET_PATH')
    if not os.path.exists(input_path):
        raise typer.BadParameter("Dataset not found. Please set the DATASET_AFFECT_NET_PATH environment variable.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    labels_path = os.path.join(input_path, "labels.csv")
    labels_output_path = os.path.join(output_dir, "labels.csv")
    os.system(f"cp {labels_path} {labels_output_path}")
    print(f"Copied labels to {labels_output_path}.")

    clip_affect_net_faces(input_path, output_dir)
    print (f"Clipped images saved to {output_dir}.")


if __name__ == "__main__":
    app()
