import numpy as np
import json
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import typer
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import scipy
from scipy.stats import entropy
from scipy.special import kl_div
from typing import List
import torch 
import torch.nn.functional as F

from inference import apply_model
from utils import label_from_path, LABEL_TO_STR

load_dotenv()
app = typer.Typer()


### Code for extracting activation values (training & validation data)

def get_activation_values(model_name, data_path):
    '''Takes a model name as str and a data path as str and 
    returns a nested dictionary of activation values grouped by true label.
    For each model_name there are nested dictionaries for each label
    with lists of activation values for each sample.'''
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
    
    return activation_values_dict


### Code for distributions

# conversion needed to save numpy arrays in json
def convert_np_arrays_to_lists(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_np_arrays_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_arrays_to_lists(item) for item in obj]
    else:
        return obj
    

def safe_file_path(dict, output_path, file_name='activation_values.json'):
    file_path = os.path.join(output_path, file_name)
    with open(file_path, 'w') as f:
        json.dump(convert_np_arrays_to_lists(dict), f)
        # values will be saved in a json file (.env file needs to be updated with path to folder)


# get average of activation values for each label (adding constant for stability)
def get_avg_activation_values_by_index(activation_values_dict, constant=0):
    '''Takes a nested dictionary of activation values and returns a nested 
    dictionary of average activation values by list index for each label. 
    The constant is added for stability.'''
    avg_activation_values_dict = {}
    for model_name, label_dict in activation_values_dict.items():
        avg_activation_values_dict[model_name] = {}
        for label, values in label_dict.items():
            # transpose the list of lists to claculate avg by index
            transposed_values = list(map(list, zip(*values)))
            # constant is added for stability
            avg_activation_values_dict[model_name][label] = [np.mean(np.array(val) + constant)
                                                              for val in transposed_values]
    return avg_activation_values_dict


### Convert activation values to probaility distributions

def softmax(x, beta = 1.0):
    '''Compute softmax values for each sets of scores in x 
    and tunes the distribution depending on beta value.'''
    # beta is higher for focus on certain values, lower for more uniform distribution
    x = x.astype(float)  # convert the inner values to floats to avoid numpy error
    e_x = np.exp(x * beta - np.max(x))  
    # subtract max(x) for numerical stability, multiply with beta for temperature scaling
    return e_x / e_x.sum(axis=0)


def get_prob_distributions(avg_activation_values_dict, beta=1.0):
    '''Takes a nested dictionary of average activation values and returns a nested 
    dictionary of probability distributions for each label.'''
    distributions_dict = {}

    for model_name, label_dict in avg_activation_values_dict.items():
        distributions_dict[model_name] = {}
        for label, values in label_dict.items():
            if all(isinstance(item, (int, float)) for item in values):
                flat_values = np.array(values)
            else:
                # flatten the list and convert to numpy array
                flat_values = np.array([item for sublist in values for item in sublist])
            distributions_dict[model_name][label] = softmax(flat_values, beta=beta)

    return distributions_dict


### Code for plotting distributions

def plot_distributions(prob_distributions_dict, output_path):
    '''Takes a nested dictionary of probability distributions and saves a plot of each distribution.'''
    for model_name, label_dict in prob_distributions_dict.items():
        for label, distribution in label_dict.items():
            plt.plot(distribution)
            plt.title(f'Distribution for label {label} in model {model_name}')
            plt.xlabel('Index')
            plt.ylabel('Probability')
            plt.savefig(os.path.join(output_path, f'distribution_{model_name}_{label}.png'))
            plt.close()


### Getting all true value distributions 
            
def get_true_value_distributions(activation_values_dict, output_path, beta=1.0, constant=0):
    '''Takes a nested dictionary of activation values and an output path as str, 
    calculates the average activation values by index, converts these to probability distributions, 
    and saves a plot of each distribution.''' 
    avg_activation_values_dict = get_avg_activation_values_by_index(activation_values_dict,
                                                                     constant=constant)
    
    safe_file_path(avg_activation_values_dict, output_path, 'avg_activation_values.json')
    
    distributions_dict = get_prob_distributions(avg_activation_values_dict, beta=beta)
    
    safe_file_path(distributions_dict, output_path, 'distributions.json')
    plot_distributions(distributions_dict, output_path)

    return distributions_dict


### First Softmax then average with threshold

def get_avg_softmax_activation_values(activation_values_dict, output_path, beta=1.0, constant=0, threshold=None):
    '''Takes a nested dictionary of activation values and an output path as str, 
    applies softmax to each list of activation values, calculates the average activation values by index, 
    and saves the results. Returns only those samples where the highest softmax value is below the given threshold.'''
    
    avg_softmax_activation_values_dict = {}
    for model_name, label_dict in activation_values_dict.items():
        avg_softmax_activation_values_dict[model_name] = {}
        for label, values in label_dict.items():
            # Apply softmax to each list of activation values
            softmax_values = [softmax(np.array(val) + constant, beta=beta) for val in values]
            # Filter out samples where the highest softmax value is above the threshold
            if threshold is not None:
                softmax_values = [val for val in softmax_values if max(val) < threshold]
            # Transpose the list of lists to calculate avg by index
            transposed_values = list(map(list, zip(*softmax_values)))
            avg_softmax_activation_values_dict[model_name][label] = [np.mean(val) for val in transposed_values]
    
    safe_file_path(avg_softmax_activation_values_dict, output_path, 'avg_softmax_activation_values.json')
    plot_distributions(avg_softmax_activation_values_dict, output_path)

    return avg_softmax_activation_values_dict


### Code for extracting distribution of single sample

def get_inf_distributions(model_name, data_path, beta=1.0, constant=0.0):
    '''Takes a model name and a data path, applies the model to the data,
    and returns a softmax distribution of the activation values for each sample.'''
    model_id, activation_values_df = apply_model(model_name, data_path)

    if not all(isinstance(label, str) for label in activation_values_df['file']):
        raise ValueError("The 'file' column should contain strings.")

    # Sort the DataFrame by the 'file' column
    activation_values_df = activation_values_df.sort_values(by='file')

    labels = activation_values_df['file'].values.tolist()

    # file column not needed anymore
    columns = [col for col in activation_values_df.columns if col != 'file']

    if not all(activation_values_df[col].dtype in ['int64', 'float64'] for col in columns):
        raise ValueError("The columns other than 'file' should contain numeric values.")

    inf_distributions = {}

    # get the distribution for each row
    for i, row in activation_values_df.iterrows():
        activation_values = row[columns].values.ravel()
        activation_values = np.nan_to_num(activation_values)
        # add the constant to each activation value
        activation_values += constant
        # flatten the activation_values array
        activation_values = activation_values.flatten()
        distribution = softmax(activation_values, beta=beta)
        inf_distributions[labels[i]] = distribution.tolist()

    return inf_distributions


### Code for calculating KL divergence

def kl_divergence(inference_distribution, distributions_dict):
    '''Takes a probability distribution and a dictionary of distributions,
    compares the input distribution to each distribution in the dictionary using KL divergence,
    and returns the results as a list.'''
    kl_divergences = []
    for model_name, distributions in distributions_dict.items():
        for label, distribution in distributions.items():
            kl_divergences.append((model_name, label, kl_div(inference_distribution, distribution)))

    return kl_divergences


def load_json(output_path, file=''):
    '''Loads a json file and returns the data.'''
    if file == '':
        raise ValueError('No file specified.')
    
    with open(os.path.join(output_path, file), 'r') as f:
        data = json.load(f)
    return data


def kl_divergence_pytorch(inf_distribution, true_distributions):
    kl_divergences = {}
    inf_distribution = torch.tensor(inf_distribution).log().view(1, -1)  # Take the logarithm of inf_distribution

    for model_name, distributions in true_distributions.items():
        for label, true_distribution in distributions.items():
            true_distribution = torch.tensor(true_distribution).view(1, -1)

            if inf_distribution.shape != true_distribution.shape:
                print(f"Skipping label {label} in model {model_name} because inf_distribution and true_distribution have different sizes.")
                continue

            kl_divergence = F.kl_div(inf_distribution, true_distribution, reduction='batchmean')  # Both distributions are already softmaxed
            
            kl_divergences[label] = kl_divergence.item()  
    return kl_divergences


def filter_distributions(inf_distributions, threshold):
    '''Takes a dictionary of inference distributions and a threshold.
    Separates the distributions into those above and below the threshold.
    Returns two dictionaries: one for the distributions above the threshold and one for the distributions below the threshold.'''
    
    if threshold is None:
        return {}, inf_distributions

    above_threshold = {}
    below_threshold = {}
    for label, distribution in inf_distributions.items():
        if max(distribution) >= threshold:
            above_threshold[label] = distribution
        else:
            below_threshold[label] = distribution

    return above_threshold, below_threshold


def get_kl_results(model_name, data_path: str = os.getenv('DATASET_TEST_PATH'), 
                            dist_path: str = os.getenv('ACTIVATION_VALUES_PATH'),
                            beta=1.0, constant=0.0, threshold=None):
    true_distributions = load_json(dist_path, 'distributions.json')
    inf_distributions = get_inf_distributions(model_name, data_path, beta=beta, constant=constant)
    above_threshold_dict, below_threshold = filter_distributions(inf_distributions, threshold)

    kl_results = []
    below_threshold_labels = []

    for label, inf_distribution in below_threshold.items():
        kl_divergences = kl_divergence_pytorch(inf_distribution, true_distributions)
        kl_divergences['distribution_name'] = label
        kl_results.append(kl_divergences)
        below_threshold_labels.append(label.split('_')[-1])

    kl_divergences_df = pd.DataFrame(kl_results)

    # Extract the labels from the above_threshold dictionary
    above_threshold_labels = list(above_threshold_dict.keys())

    # Convert the above_threshold dictionary to a DataFrame
    above_threshold_df = pd.DataFrame.from_dict(above_threshold_dict, orient='index')

    return above_threshold_df, kl_divergences_df, below_threshold_labels, above_threshold_labels


def kl_confusion_matrix(kl_results_df: pd.DataFrame):
    """Return the confusion matrix."""
    LABEL_TO_STR = {i: str(i) for i in range(len(kl_results_df.columns) - 1)}
    LABEL_TO_STR[len(LABEL_TO_STR)] = 'inf'

    matrix = np.zeros((len(LABEL_TO_STR), len(LABEL_TO_STR)))

    for index, row in kl_results_df.iterrows():
        label = int(row['Label'])
        kl_divergences = row['KL-Divergence']
        predictions = kl_divergences.argsort()[::-1]
        if predictions[0] in LABEL_TO_STR:
            if LABEL_TO_STR[predictions[0]] == 'inf':
                # Handle the case where LABEL_TO_STR[predictions[0]] is 'inf'
                # Replace this line with the appropriate action
                pass
            else:
                matrix[label, int(LABEL_TO_STR[predictions[0]])] += 1

    return pd.DataFrame(matrix, index=LABEL_TO_STR.values(), columns=LABEL_TO_STR.values())


def kl_divergence_accuracies(kl_results_df, above_threshold_df, below_threshold_labels, above_threshold_labels):
    top1 = 0.0
    top3 = 0.0
    top1_labels = []

    labels = below_threshold_labels + above_threshold_labels

    for i, row in kl_results_df.iterrows():
        true_label = labels[i].replace('.jpg', '')
        row = row[row.apply(lambda x: isinstance(x, (int, float)))]
        if row.empty:
            continue

        pred_labels = row.sort_values().index.tolist()
        pred_labels = [LABEL_TO_STR[int(label)] for label in pred_labels]
        top1_labels.append(pred_labels[0])

        if true_label == pred_labels[0]:
            top1 += 1
        if true_label in pred_labels[:3]:  # Check if true_label is in the top 3 predictions
            top3 += 1

    for i, row in above_threshold_df.iterrows():
        # Use i directly as the true label for a row in above_threshold_df
        true_label = i.replace('.jpg', '')
        row = row[row.apply(lambda x: isinstance(x, (int, float)))]
        if row.empty:
            continue

        pred_labels = row.sort_values(ascending=False).index.tolist()
        pred_label = pred_labels[0]
        pred_label = LABEL_TO_STR[int(pred_label)]
        top1_labels.append(pred_label)

        if true_label == pred_label:
            top1 += 1
        if true_label in pred_labels[:3]:  # Check if true_label is in the top 3 predictions
            top3 += 1

    top1 /= len(labels)
    top3 /= len(labels)
    return top1, top3, top1_labels


def generate_confusion_matrix(below_threshold_labels, above_threshold_labels, top1_labels):
    # Concatenate the below_threshold_labels and above_threshold_labels to form true_labels
    true_labels = below_threshold_labels + above_threshold_labels

    # Remove the '.jpg' extension from the true labels
    true_labels = [label.replace('.jpg', '') for label in true_labels]

    cm = confusion_matrix(true_labels, top1_labels)

    return cm



