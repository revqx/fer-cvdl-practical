import numpy as np
import json
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import typer
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch 
import torch.nn.functional as F

from inference import apply_model
from utils import label_from_path, LABEL_TO_STR

load_dotenv()
app = typer.Typer()


def get_activation_values(model_name, data_path, output_path):
    """Takes a model name as str and a data path as str and 
    returns a nested dictionary of activation values grouped by true label.
    For each model_name there are nested dictionaries for each label
    with lists of activation values for each sample."""
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
        
    safe_file_path(activation_values_dict, output_path, 'activation_values.json')
    
    return activation_values_dict


# Conversion needed to save numpy arrays in json
def convert_np_arrays_to_lists(obj):
    """Converts numpy arrays to lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_np_arrays_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_arrays_to_lists(item) for item in obj]
    else:
        return obj
    

def safe_file_path(dict, output_path, file_name='activation_values.json'):
    """Takes a dictionary and an output path as str,
    saves the dictionary as a json file in the output path."""
    file_path = os.path.join(output_path, file_name)
    with open(file_path, 'w') as f:
        json.dump(convert_np_arrays_to_lists(dict), f)
        # Values will be saved in a json file (.env file needs to be updated with path to folder)


def softmax(x, temperature = 1.0):
    """Compute softmax values for each sets of scores in x 
    and tunes the distribution depending on temperature value."""
    # temperature is higher for focus on certain values, lower for more uniform distribution
    x = x.astype(float)  # convert the inner values to floats to avoid numpy error
    e_x = np.exp(x / temperature - np.max(x))  
    # Subtract max(x) for numerical stability, multiply with temperature for temperature scaling
    return e_x / e_x.sum(axis=0)


def plot_distributions(prob_distributions_dict, output_path):
    """Takes a nested dictionary of probability distributions and saves a plot of each distribution."""
    for model_name, label_dict in prob_distributions_dict.items():
        for label, distribution in label_dict.items():
            plt.plot(distribution)
            plt.title(f'Distribution for label {label} in model {model_name}')
            plt.xlabel('Index')
            plt.ylabel('Probability')
            plt.savefig(os.path.join(output_path, f'distribution_{model_name}_{label}.png'))
            plt.close()


def get_avg_softmax_activation_values(activation_values_dict, output_path, temperature=1.0, constant=0, threshold=None):
    """Takes a nested dictionary of activation values and an output path as str, 
    applies softmax to each list of activation values, calculates the average activation values by index, 
    and saves the results. Returns only those samples where the highest softmax value is below the given threshold."""
    
    avg_softmax_activation_values_dict = {}
    for model_name, label_dict in activation_values_dict.items():
        avg_softmax_activation_values_dict[model_name] = {}
        for label, values in label_dict.items():
            adjusted_values = [np.array(val) + constant for val in values]
            if threshold is not None:
                adjusted_values = [val for val in adjusted_values if max(val) < threshold]

            softmax_values = [softmax(val, temperature=temperature) for val in adjusted_values]

            for softmax_val in softmax_values:
                assert np.isclose(sum(softmax_val), 1, atol=1e-6), "Softmax distribution does not sum to 1"

            transposed_values = list(map(list, zip(*softmax_values)))
            avg_softmax_activation_values_dict[model_name][label] = [np.mean(val) for val in transposed_values]
    
    safe_file_path(avg_softmax_activation_values_dict, output_path, 'avg_softmax_activation_values.json')
    plot_distributions(avg_softmax_activation_values_dict, output_path)

    return avg_softmax_activation_values_dict


def get_inf_distributions(model_name, data_path, output_path, temperature=1.0, constant=0.0, threshold=None):
    """Takes a model name and a data path, applies the model to the data,
    and returns a softmax distribution of the activation values for each sample."""
    model_id, activation_values_df = apply_model(model_name, data_path)

    if not all(isinstance(label, str) for label in activation_values_df['file']):
        raise ValueError("The 'file' column should contain strings.")

    activation_values_df = activation_values_df.sort_values(by='file')
    labels = activation_values_df['file'].values.tolist()

    # File column not needed anymore
    columns = [col for col in activation_values_df.columns if col != 'file']

    if not all(activation_values_df[col].dtype in ['int64', 'float64'] for col in columns):
        raise ValueError("The columns other than 'file' should contain numeric values.")

    inf_distributions = {}
    above_threshold = {}
    below_threshold = {}

    # Get the distribution for each row
    for i, row in activation_values_df.iterrows():
        activation_values = row[columns].values.ravel()
        activation_values = np.nan_to_num(activation_values)
        # Add the constant to each activation value
        adjusted_activation_values = activation_values + constant
        # Flatten the activation_values array
        adjusted_activation_values = adjusted_activation_values.flatten()
        distribution = softmax(adjusted_activation_values, temperature=temperature)
        inf_distributions[labels[i]] = distribution.tolist()

        # Threshold check
        if threshold is not None:
            if max(adjusted_activation_values) >= threshold:
                above_threshold[labels[i]] = distribution.tolist()
            else:
                below_threshold[labels[i]] = distribution.tolist()

    safe_file_path(inf_distributions, output_path, 'inf_distributions.json')

    return above_threshold, below_threshold


def load_json(output_path, file=''):
    """Loads a json file and returns the data."""
    if file == '':
        raise ValueError('No file specified.')
    
    with open(os.path.join(output_path, file), 'r') as f:
        data = json.load(f)
    return data


def kl_divergence_pytorch(inf_distribution, true_distributions):
    """Takes a probability distribution and a dictionary of distributions,
    compares the input distribution to each distribution in the dictionary using KL divergence,
    and returns the results as a dictionary."""
    kl_divergences = {}
    # Pytorch kl_div function requires input to be log probabilities and tensors
    inf_distribution = torch.tensor(inf_distribution).log().view(1, -1)  # Log of inf_distribution

    for model_name, distributions in true_distributions.items():
        for label, true_distribution in distributions.items():
            true_distribution = torch.tensor(true_distribution).view(1, -1)

            if inf_distribution.shape != true_distribution.shape:
                print(f"Skipping label {label} in model {model_name} because inf_distribution and true_distribution have different sizes.")
                continue

            kl_divergence = F.kl_div(inf_distribution, true_distribution, reduction='batchmean')
            
            kl_divergences[label] = kl_divergence.item()  

    return kl_divergences


def get_kl_results(model_name, output_path, data_path: str = os.getenv('DATASET_TEST_PATH'), 
                            dist_path: str = os.getenv('ACTIVATION_VALUES_PATH'),
                            temperature=1.0, constant=0.0, threshold=None):
    """Takes a model name, a data path, and a distribution path,
    calculates the KL divergence between the inference distributions and the true distributions,
    and returns the results as a dataframe."""
    
    true_distributions = load_json(dist_path, 'avg_softmax_activation_values.json')
    above_threshold, below_threshold = get_inf_distributions(model_name, data_path, output_path,
                                                                temperature=temperature, constant=constant, threshold=threshold)

    kl_results = []
    below_threshold_labels = []

    for label, inf_distribution in below_threshold.items():
        kl_divergences = kl_divergence_pytorch(inf_distribution, true_distributions)
        kl_divergences['distribution_name'] = label
        kl_results.append(kl_divergences)
        below_threshold_labels.append(label.split('_')[-1])

    kl_divergences_df = pd.DataFrame(kl_results)

    # Get true_labels from the above_threshold dictionary
    above_threshold_labels = list(above_threshold.keys())
    above_threshold_df = pd.DataFrame.from_dict(above_threshold, orient='index')

    return above_threshold_df, kl_divergences_df, below_threshold_labels, above_threshold_labels


def kl_divergence_accuracies(kl_results_df, above_threshold_df, below_threshold_labels, above_threshold_labels):
    """Takes a dataframe of KL divergence results and a dataframe of above threshold results,
    calculates the top 1 and top 3 accuracies, and returns the results as floats."""
    top1 = 0.0
    top3 = 0.0
    top1_labels = []
    true_labels = []
    debug_dists = []

    labels = below_threshold_labels + above_threshold_labels

    for i, row in kl_results_df.iterrows():
        true_label = labels[i].replace('.jpg', '')
        true_labels.append(true_label)
        row = row[row.apply(lambda x: isinstance(x, (int, float)))]
        if row.empty:
            continue
        # Sort by min kl-divergence value
        pred_labels = row.sort_values().index.tolist()
        debug_dists.append(row.values.tolist()[:6])
        pred_labels = [LABEL_TO_STR[int(label)] for label in pred_labels]
        top1_labels.append(pred_labels[0])

        if true_label == pred_labels[0]:
            top1 += 1
        if true_label in pred_labels[:3]:  
            top3 += 1

    for i, row in above_threshold_df.iterrows():
        filename = os.path.basename(i)
        true_label = filename.split('_')[1].replace('.jpg', '')
        true_labels.append(true_label)
        row = row[row.apply(lambda x: isinstance(x, (int, float)))]
        if row.empty:
            continue
        # Sort by max softmax value
        pred_labels = row.sort_values(ascending=False).index.tolist()
        pred_labels = [LABEL_TO_STR[int(label)] for label in pred_labels]
        top1_labels.append(pred_labels[0])

        if true_label == pred_labels[0]:
            top1 += 1
        if true_label in pred_labels[:3]: 
            top3 += 1

    top1 /= len(labels)
    top3 /= len(labels)
    return top1, top3, top1_labels, true_labels, debug_dists


def generate_confusion_matrix(true_labels, top1_labels):
    """Takes a list of true labels and a list of predicted labels,
    calculates the confusion matrix, and returns the results as a numpy array."""

    cm = confusion_matrix(true_labels, top1_labels)

    return cm
