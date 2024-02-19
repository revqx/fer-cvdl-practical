import os
from datetime import datetime
import json

import cv2

import numpy as np
import typer
import wandb
from dotenv import load_dotenv

from explain import pca_graph, explain_with_method, explain_all

from analyze import accuracies, confusion_matrix, analyze_run_and_upload
from clip_affect_net import clip_affect_net_faces
from distribution import get_activation_values, get_kl_results, kl_divergence_accuracies, \
    get_avg_softmax_activation_values
from ensemble import ensemble_results, save_confusion_matrix_as_heatmap
from inference import apply_model, load_model_and_preprocessing
from sweep import get_sweep_config
from train import train_model
from video_prediction import make_video_prediction

load_dotenv()
app = typer.Typer()

TRAIN_CONFIG = {
    "model_name": "ConvModel1",
    # Options: LeNet, ResNet{18, 50}, EmotionModel2, CustomEmotionModel{3, 4, 5}, MobileNetV2
    "model_description": "",
    "pretrained_model": "",  # Options: model_id, model_name (for better wandb logging, use the model id)
    "train_data": "AffectNet",  # Options: RAF-DB, AffectNet
    "preprocessing": "ImageNetNormalization",  # Options: ImageNetNormalization, Grayscale
    "augmentations": "HorizontalFlip, RandomRotation, RandomCrop, TrivialAugmentWide, TrivialAugmentWide",
    # Options: "HorizontalFlip", "RandomRotation", "RandomCrop", "TrivialAugmentWide", "RandAugment"
    "validation_split": 0.1,
    "learning_rate": 0.0001,
    "batch_size": 32,
    "sampler": "uniform",  # Options: uniform
    "scheduler": "StepLR",  # Options: ReduceLROnPlateau, StepLR
    "ReduceLROnPlateau_factor": 0.1,
    "ReduceLROnPlateau_patience": 2,
    "StepLR_decay_rate": 0.95,
    "loss_function": "CrossEntropyLoss",  # Options: CrossEntropyLoss
    "class_weight_adjustments": [1, 1, 1, 1, 1, 1],  # Only applied if scheduler is "uniform"
    "optimizer": "Adam",  # Options: Adam, SGD
    "device": "cuda:0",  # Options: cuda, cpu, mps
    "DynamicModel_hidden_layers": 1,
    "DynamicModel_hidden_dropout": 0.2,
    "max_epochs": "10",
    "early_stopping_patience": "5",
}

# In case you want to create an ensemble model, add the model names/id here
# Current best ensemble with 82,16 %Top1
# ["8cp5wrtr_1", "zpwmo75q", "zpwmo75q", "npl99ug4", "h1zooiju", "1p4v64b3", "1eq7h5pb"]
ENSEMBLE_MODELS = ["8cp5wrtr_1", "zpwmo75q", "zpwmo75q", "npl99ug4", "h1zooiju", "1p4v64b3", "1eq7h5pb"]


@app.command()
def train(offline: bool = False, sweep: bool = False):
    # check if validation path is valid
    if not os.path.exists(os.getenv("DATASET_TEST_PATH")):
        raise FileNotFoundError(f"Directory {os.getenv('DATASET_TEST_PATH')} not found. "
                                f"Could not load validation dataset.")

    # disable wandb if offline
    os.environ["WANDB_MODE"] = "offline" if offline else "online"

    config = TRAIN_CONFIG
    with wandb.init(config=config, project="cvdl", entity=os.getenv("WANDB_ENTITY")):
        # overwrite config with wandb config if sweep run
        if sweep:
            for key in wandb.config.as_dict():
                config[key] = wandb.config.as_dict().get(key)

        model = train_model(config)
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}"
              f" (Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)})")
        # print(model.summary())

        # test model and upload results to wandb if not a sweep run
        if not sweep:
            analyze_run_and_upload(config["model_name"])


@app.command()
def inference(model_name: str, data_path: str, output_path: str):
    model_id, results = apply_model(model_name, data_path)
    # create name from model_name and timestamp and input files
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    folder_name = data_path.split("/")[-1]
    output_file_name = f"{folder_name}-{timestamp}-{model_id}.csv"
    print(f"Writing results to {output_file_name}.")
    results.to_csv(os.path.join(output_path, output_file_name), index=False)


@app.command()
def analyze(model_name: str, data_path: str = os.getenv("DATASET_TEST_PATH")):
    model_id, results = apply_model(model_name, data_path)
    top_n = accuracies(results)
    conf_matrix = confusion_matrix(results)
    print(conf_matrix)
    print(top_n)

    return model_id, top_n, conf_matrix


@app.command()
def demo(model_name: str, webcam: bool = False, cam_id: int = 0,
         input_video: str = "",
         show_processing: bool = True, explainability: bool = False,
         landmarks: bool = False, info: bool = True, codec: str = "XVID",
         output_ext: str = "avi"):
    if not webcam and input_video.strip() == "":
        raise typer.BadParameter("Please specify an input video when not using the camera.")

    if webcam:
        cap = cv2.VideoCapture(cam_id)
        is_valid = cap.isOpened()
        cap.release()
        if not is_valid:
            raise typer.BadParameter("Please specify a valid camera id")

    if not os.path.exists("models/version-RFB-320.onnx"):
        raise FileNotFoundError("version-RFB-320.onnx not found. "
                                "Please download the file from "
                                "https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/models/onnx/version-RFB-320.onnx "
                                "and put it to the /models directory.")

    if landmarks and not os.path.exists("models/shape_predictor_68_face_landmarks.dat"):
        raise FileNotFoundError("shape_predictor_68_face_landmarks.dat not found. "
                                "Please download the file from "
                                "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 "
                                "and extract it to the /models directory.")

    output_dir = os.getenv("VIDEO_OUTPUT_PATH")
    if not os.path.exists(output_dir) and not webcam:
        os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_file = os.path.join(output_dir, f"{model_name}-{timestamp}.{output_ext}")
    make_video_prediction(model_name, webcam, cam_id, input_video, output_file, show_processing, explainability, landmarks, info, codec)


@app.command()
def clip(output_dir: str = "data/clipped_affect_net", use_rafdb_format: bool = False):
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
    """Takes the data_path and applies the ensemble of models to it.
    Returns the top1 and top3 accuracies and the confusion matrix."""
    model_ids = ENSEMBLE_MODELS
    ensemble_results_df = ensemble_results(model_ids, data_path)

    top_n = accuracies(ensemble_results_df)
    conf_matrix = confusion_matrix(ensemble_results_df)

    labels = [col for col in ensemble_results_df.columns if col != 'path']
    heatmap_filename = "ensemble-confusion-matrix-dark.png"
    save_confusion_matrix_as_heatmap(conf_matrix, labels, heatmap_filename)

    print(conf_matrix)
    print(top_n)


@app.command()
def sweep(sweep_id: str = "", count: int = 40):
    sweep_config = get_sweep_config()
    entity = os.getenv("WANDB_ENTITY")

    if sweep_id == "":
        sweep_id = wandb.sweep(sweep_config, project="cvdl", entity=entity)

    wandb.agent(sweep_id, function=lambda: train(sweep=True), project="cvdl", count=count, entity=entity)


@app.command()
def true_value_distributions(model_name: str, data_path: str = os.getenv("DATASET_RAF_DB_PATH"),
                             output_path = os.getenv("ACTIVATION_VALUES_PATH")):
    """Takes the model_name and data_path, loads the activation values and calculates the true value distributions."""
    activation_values_dict = get_activation_values(model_name, data_path, output_path)
    get_avg_softmax_activation_values(activation_values_dict, output_path,
                                      constant=20, temperature=0.5, threshold=24)

@app.command()
def kl_analyze(model_name: str, data_path: str = os.getenv("DATASET_TEST_PATH")):
    """Takes the model_name and data_path, loads the true value distributions and calculates the kl-divergences.
    Returns the top1 and top3 accuracies and the confusion matrix."""
    output_path = os.getenv("ACTIVATION_VALUES_PATH")
    above_df, kl_divergence_df, be_labels, ab_labels = get_kl_results(model_name, output_path, data_path, constant=20,
                                                                      temperature=0.5, threshold=24)
    print(len(ab_labels))  # outputs number of samples above threshold
    top1, top3, pred_labels, true_labels, _ = kl_divergence_accuracies(kl_divergence_df, above_df, be_labels, ab_labels)
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    print(conf_matrix)
    print("Top 1 accuracy: ", top1, "Top 3 accuracy: ", top3)


@app.command()
def retrieve_val(run_dataset_version: str):
    """Takes the run_dataset_version (logged in wandb under overview -> artifact outputs)
      and retrieves the indices of the validation set as an artifact from wandb."""
    entity = os.getenv("WANDB_ENTITY")
    project = "cvdl"
    api = wandb.Api()

    artifact_name = f"{entity}/{project}/{run_dataset_version}"
    artifact = api.artifact(artifact_name, type='dataset')
    artifact_dir = artifact.download()
    val_indices = np.load(os.path.join(artifact_dir, 'val_indices.npy'))

    print(val_indices[0])

    return val_indices


@app.command()
def explain(model_name: str, method: str = 'gradcam', window: int = 8, data_path: str = os.getenv("DATASET_TEST_PATH"),
            examples: int = 5, random: bool = True, path_contains: str = "", save_path: str = None):
    model_id, model, preprocessing = load_model_and_preprocessing(model_name)
    explain_with_method(model, method, data_path, examples=examples, random=random,
                        path_contains=path_contains, save_path=save_path, window_size=(window, window))


@app.command()
def explain_image(model_name: str, window: int = 8, data_path: str = os.getenv("DATASET_TEST_PATH"), path_contains: str = "", save_path: str = None):
    model_id, model, preprocessing = load_model_and_preprocessing(model_name)
    explain_all(model, data_path, path_contains=path_contains, save_path=save_path, window_size=(window, window))


@app.command()
def pca(model_name: str, data_path: str = os.getenv("DATASET_TEST_PATH"), softmax: bool = False):
    model_id, results = apply_model(model_name, data_path)
    pca_graph(model_id, results, softmax=softmax)


if __name__ == "__main__":
    app()
