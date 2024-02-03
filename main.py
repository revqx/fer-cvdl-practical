import json
import os
from datetime import datetime
import json
import cv2

import typer
import wandb
from dotenv import load_dotenv

from explain import pca_graph
from utils import label_from_path

from gradcam import grad_cam
from analyze import accuracies, confusion_matrix, analyze_run_and_upload
from clip_affect_net import clip_affect_net_faces
from ensemble import ensemble_results
from inference import apply_model, load_model_and_preprocessing
from sweeps import get_sweep_config, train_sweep
from train import train_model
from utils import label_from_path
from video_prediction import make_video_prediction

load_dotenv()
app = typer.Typer()

# This should not be changed unless the current run is better than the current best
# See CUSTOM_TRAIN_CONFIG for config options
CURRENT_BEST_TRAIN_CONFIG = {
    "model_name": "CustomEmotionModel3",
    "model_description": "",
    "pretrained_model": "",
    "train_data": "RAF-DB",
    "preprocessing": "ImageNetNormalization",
    "augmentations": "HorizontalFlip, RandomRotation, RandomCrop, TrivialAugmentWide, TrivialAugmentWide",
    "validation_split": 0.1,
    "learning_rate": 0.001,
    "epochs": 20,
    "batch_size": 32,
    "sampler": "uniform",
    "scheduler": "ReduceLROnPlateau",
    "ReduceLROnPlateau_factor": 0.1,
    "ReduceLROnPlateau_patience": 5,
    "StepLR_decay_rate": 0.95,
    "loss_function": "CrossEntropyLoss",
    "class_weight_adjustments": [1, 1, 1, 1, 1, 1],
    "optimizer": "Adam",
    "device": "mps"
}

# If you want to use a custom config, change this one as you like
CUSTOM_TRAIN_CONFIG = {
    "model_name": "CustomEmotionModel3",
    # Options: LeNet, ResNet{18, 50}, EmotionModel2, CustomEmotionModel{3, 4, 5}, MobileNetV2
    "model_description": "",
    "pretrained_model": "p28ita7r",  # Options: model_id, model_name (for better wandb logging, use the model id)
    "train_data": "RAF-DB",  # Options: AffectNet, RAF-DB
    "preprocessing": "ImageNetNormalization",  # Options: ImageNetNormalization, Grayscale
    "augmentations": "HorizontalFlip, RandomRotation, RandomCrop, TrivialAugmentWide, TrivialAugmentWide",
    # Options: "HorizontalFlip", "RandomRotation", "RandomCrop", "TrivialAugmentWide", "RandAugment"
    "validation_split": 0.1,
    "learning_rate": 0.001,
    "epochs": 2,
    "batch_size": 32,
    "sampler": "uniform",  # Options: uniform
    "scheduler": "StepLR",  # Options: ReduceLROnPlateau, StepLR
    "ReduceLROnPlateau_factor": 0.1,
    "ReduceLROnPlateau_patience": 5,
    "StepLR_decay_rate": 0.95,
    "loss_function": "CrossEntropyLoss",  # Options: CrossEntropyLoss
    "class_weight_adjustments": [1, 1, 1, 1, 1, 1],
    "optimizer": "Adam",  # Options: Adam, SGD
    "device": "mps"  # Options: cuda, cpu, mps
}

# In case you want to create an ensemble model, add the model names/id here
ENSEMBLE_MODELS = ["h8txabjg", "odyx0ott", "8uu89woq"]


@app.command()
def train(offline: bool = False):
    # merge default and custom config
    config = CURRENT_BEST_TRAIN_CONFIG | CUSTOM_TRAIN_CONFIG
    # check if validation path is valid
    if not os.path.exists(os.getenv("DATASET_TEST_PATH")):
        raise FileNotFoundError(f"Directory {os.getenv('DATASET_TEST_PATH')} not found. "
                                f"Could not load validation dataset.")

    # disable wandb if offline
    os.environ["WANDB_MODE"] = "offline" if offline else "online"
    wandb.init(project="cvdl", config=config)
    train_model(config)

    # test model on validation data
    analyze_run_and_upload(config["model_name"])


@app.command()
def inference(model_name: str, data_path: str, output_path: str):
    model_id, results = apply_model(model_name, data_path)
    # create name from model_name and timestamp and input files
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    folder_name = data_path.split("/")[-1]
    output_file_name = f"{folder_name}-{timestamp}-{model_id}.csv"
    print(f"Writing results to {output_file_name}.")
    results.to_csv(os.path.join(output_path, output_file_name), idx=False)


@app.command()
def analyze(model_name: str, data_path: str = os.getenv("DATASET_TEST_PATH")):
    model_id, results = apply_model(model_name, data_path)
    top_n = accuracies(results)
    conf_matrix = confusion_matrix(results)
    print(conf_matrix)
    print(top_n)


@app.command()
def demo(model_name: str, save: bool = False, webcam: bool = False, cam_id: int = 0, input_file: str = "",
         show_processing: bool = True, explanation: bool = False, details: bool = False, info: bool = True, hog: bool = False):
    if not webcam and input_file.strip() == "":
        raise typer.BadParameter("Please specify a video input when not using the camera.")

    if webcam:
        cap = cv2.VideoCapture(cam_id)
        is_valid = cap.isOpened()
        cap.release()
        if not is_valid:
            raise typer.BadParameter("Please specify a valid camera id")

    output_dir = os.getenv("VIDEO_OUTPUT_PATH")
    if not os.path.exists(output_dir) and save:
        os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_file = os.path.join(output_dir, f"{model_name}-{timestamp}.avi")
    make_video_prediction(model_name, save, webcam, cam_id, input_file, output_file, show_processing, explanation, details, info, hog)


@app.command()
def clipped(output_dir: str = "data/clipped_affect_net", use_rafdb_format: bool = False):
    input_path = os.getenv("DATASET_AFFECT_NET_PATH")
    if not os.path.exists(input_path):
        raise typer.BadParameter("Dataset not found. Please set the DATASET_AFFECT_NET_PATH environment variable.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if not use_rafdb_format:
        labels_path = os.path.join(input_path, "labels.csv")
        labels_output_path = os.path.join(output_dir, "labels.csv")
        os.system(f"cp {labels_path} {labels_output_path}")
        print(f"Copied labels to {labels_output_path}.")

    clip_affect_net_faces(input_path, output_dir, use_rafdb_format)
    print(f"Clipped images saved to {output_dir}.")


@app.command()
def ensemble(data_path=os.getenv("DATASET_TEST_PATH")):
    model_ids = ENSEMBLE_MODELS
    ensemble_results_df = ensemble_results(model_ids, data_path)

    top_n = accuracies(ensemble_results_df)
    conf_matrix = confusion_matrix(ensemble_results_df)
    print(conf_matrix)
    print(top_n)


@app.command()
def initialize_sweep(entity: str = "your_user_name", count: int = 40):
    project = "cvdl"

    if entity == "your_user_name":
        raise ValueError("Please enter your user name.")

    sweep_config = get_sweep_config()

    sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
    wandb.agent(sweep_id, function=train_sweep, count=count)


@app.command()
def get_activation(model_name: str, data_path: str = os.getenv("DATASET_TEST_PATH"),
                     output_path: str = os.getenv("ACTIVATION_VALUES_PATH")):
    model_id, results = apply_model(model_name, data_path)
    labels = []
    activation_values_dict = {}

    for path, *values in results.values:
        label = label_from_path(path)
        labels.append(label)
        if label is None:
            raise ValueError(f"Could not find label in path {path}.")

        if model_name not in activation_values_dict:
            activation_values_dict[model_name] = {}

        if label not in activation_values_dict[model_name]:
            activation_values_dict[model_name][label] = []

        activation_values_dict[model_name][label].append(values)

    # save values locally as a json file (folder path from env file)
    with open(f"{output_path}\\activation_values.json", "w") as f:
        json.dump(activation_values_dict, f)


@app.command()
def explain(model_name: str, data_path: str = os.getenv("DATASET_VALIDATION_PATH"), examples: int = 5,
            random: bool = True, path_contains: str = "", save_path: str = None):
    model_id, model, preprocessing = load_model_and_preprocessing(model_name)
    grad_cam(model, data_path, examples=examples, random=random, path_contains=path_contains, save_path=save_path)


@app.command()
def pca(model_name: str, data_path: str = os.getenv("DATASET_VALIDATION_PATH"), softmax: bool = False):
    model_id, results = apply_model(model_name, data_path)
    pca_graph(model_id, results, softmax=softmax)


if __name__ == "__main__":
    app()
