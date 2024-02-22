def get_sweep_config(metric="val_loss", goal="minimize", method="random",
                     custom_model=False, early_terminate=False):
    sweep_config = {
        "method": method
    }

    metric = {
        "name": metric,
        "goal": goal
    }
    sweep_config["metric"] = metric

    # Parameters to sweep over (dropout not possible atm because models need custom input for dropout)
    parameters_dict = {
        "optimizer": {
            "values": ["Adam"]  # Options: Adam, SGD
        },
        "train_data": {
            "values": ["RAF-DB"]  # Options: AffectNet, RAF-DB
        },
        "batch_size": {
            "values": [32]  # defined here since log distribution causes bad comparability
        },
        "validation_split": {
            "values": [0.2]  # Once sweeped to be set as constant
        },
        "scheduler": {
            "values": ["ReduceLROnPlateau", "StepLR"]
        },
        "ReduceLROnPlateau_patience": {
            "values": [3]
        },
        "augmentations": {
            "values": [
                "HorizontalFlip, RandomRotation, RandomCrop, TrivialAugmentWide, TrivialAugmentWide",
                "HorizontalFlip, RandomRotation, RandomCrop, TrivialAugmentWide",
                "HorizontalFlip, RandomRotation, RandomCrop, RandAugment, RandAugment",
                "HorizontalFlip, RandomRotation, RandomCrop, RandAugment",
                "HorizontalFlip, RandomRotation, RandomCrop"
            ]
        },
        "pretrained_model": {
            "values": ["", "gk1oomks"]
        }
    }

    sweep_config["parameters"] = parameters_dict

    if custom_model:
        parameters_dict.update({
            "model_name": {
                "values": ["DynamicModel"]
            },
            "DynamicModel_dropout": {
                "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            },
            "DynamicModel_hidden_layers": {
                "values": [0, 1, 2, 3, 4, 5]
            }
        })
        sweep_config["parameters"] = parameters_dict
    else:
        parameters_dict.update({
            "model_name": {
                "values": ["CustomEmotionModel7"]
            }  # options: EmotionModel_2, CustomEmotionModel3, LeNet, ResNet18
        })
        sweep_config["parameters"] = parameters_dict

    if sweep_config["method"] == "grid" or sweep_config["method"] == "bayes":
        parameters_dict.update({
            "learning_rate": {
                # a flat distribution between 0 and 0.1
                "values": [0.001, 0.0005, 0.0001]
            },
            "max_epochs": {
                "values": [15]
            }
        })
    else:
        parameters_dict.update({
            "learning_rate": {
                # a flat distribution between 0 and 0.1
                #"distribution": "uniform",
                #"min": 0.00001,
                #"max": 0.001
                "values": [0.0001, 0.00001]
            },
            "max_epochs": {
                "values": [5]  # adjust to your liking (3 gives more accurate results than 1)
            }
        })
    sweep_config["parameters"] = parameters_dict

    if early_terminate:
        sweep_config["early_terminate"] = {"type": "hyperband", "min_iter": 5, "eta": 3}

    return sweep_config


def get_tune_config():
    sweep_config = {
        'method': 'grid',  # Can be 'grid', 'random', or 'bayes'
        'metric': {
            'name': 'Top 1 accuracy',
            'goal': 'maximize'
        },
        'parameters': {  # To be defined differently for grid 
            'temperature': {
                #'distribution': 'q_uniform',
                #'min': 0.5,
                #'max': 8,
                #'q': 0.5 
                'values': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 8.0]
            },
            'threshold': {
                'values': [19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 25.5, 26.0, 26.5, 27.0, 27.5]
                #'values': [20.2, 20.5, 20.8, 21.1, 21.4, 21.7, 22.0, 22.3, 22.6, 22.9, 23.2, 23.5]
            },
            'constant': {
                'value': 20
            }
        }
    }

    return sweep_config
