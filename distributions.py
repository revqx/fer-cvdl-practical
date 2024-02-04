import numpy as np
import json
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import typer
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.stats import entropy
from scipy.special import kl_div
from typing import List

from inference import apply_model
from utils import label_from_path, LABEL_TO_STR
import warnings ######## FOR DEBUGGING ########

load_dotenv()
app = typer.Typer()

### Code for extracting activation values (training & validation data)
def get_activation_values(model_name, data_path):
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


### Code for correlation metrics
def get_correlation_of_indices(activation_values_dict):
    correlation_dict = {}

    for model_name in activation_values_dict:
        correlation_dict[model_name] = {}

        for label in activation_values_dict[model_name]:
            values = np.array(activation_values_dict[model_name][label])
            correlation_matrix = np.corrcoef(values, rowvar=False)
            correlation_dict[model_name][label] = correlation_matrix.tolist()

    return correlation_dict


def get_correlation_of_emotions(correlation_dict):
    correlation_emotions_dict = {}

    for model_name in correlation_dict:
        correlation_emotions_dict[model_name] = {}

        for label in correlation_dict[model_name]:
            correlation_emotions_dict[model_name][label] = {}

            for i in range(len(correlation_dict[model_name][label])):
                correlation_emotions_dict[model_name][label][i] = {}

                for j in range(len(correlation_dict[model_name][label][i])):
                    correlation_emotions_dict[model_name][label][i][j] = {}

                    for emotion in range(6):
                        correlation_emotions_dict[model_name][label][i][j][emotion] = correlation_dict[model_name][label][i][j][emotion]

    return correlation_emotions_dict


def correlation(activation_values_dict, output_path):

    correlation_dict = get_correlation_of_indices(activation_values_dict)
    safe_file_path(correlation_dict, output_path, file_name='correlation_indices.json')

    return correlation_dict


### Code for distributions
def safe_file_path(dict, output_path, file_name='activation_values.json'):
    file_path = os.path.join(output_path, file_name)
    with open(file_path, 'w') as f:
        json.dump(convert_np_arrays_to_lists(dict), f)
        # values will be saved in a json file (.env file needs to be updated with path to folder)


def convert_np_arrays_to_lists(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_np_arrays_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_arrays_to_lists(item) for item in obj]
    else:
        return obj
    

def softmax(x):
    e_x = np.exp(x - np.max(x))  # subtract max(x) for numerical stability
    return e_x / e_x.sum(axis=0)


def activation_to_probabilities_2(activation_values_dict):
    probabilities_dict = {}

    for model_name in activation_values_dict:
        probabilities_dict[model_name] = {}

        for emotion in activation_values_dict[model_name]:
            probabilities_dict[model_name][emotion] = []

            for i in range(len(activation_values_dict[model_name][emotion])):
                activation_values = np.array(activation_values_dict[model_name][emotion][i])
                probabilities = softmax(activation_values)
                probabilities_dict[model_name][emotion].append(probabilities.tolist())

    return probabilities_dict


# Option 1: Average probabilities for each emotion across all images for each image index
def get_average_probabilities(probabilities_dict):
    average_probabilities_dict = {}
    
    for model_name in probabilities_dict:
        average_probabilities_dict[model_name] = {}

        for emotion in probabilities_dict[model_name]:
            probabilities = np.array(probabilities_dict[model_name][emotion])
            average_probabilities = np.mean(probabilities, axis=0)
            average_probabilities_dict[model_name][emotion] = average_probabilities.tolist()

    return average_probabilities_dict


def get_average_activations(activation_values_dict):
    average_activations_dict = {}
    
    for model_name in activation_values_dict:
        average_activations_dict[model_name] = {}

        for emotion in activation_values_dict[model_name]:
            activations = np.array(activation_values_dict[model_name][emotion])
            average_activations = np.mean(activations, axis=0)
            average_activations_dict[model_name][emotion] = average_activations.tolist()

    return average_activations_dict


# Option 2: Same function with median (flattens out outliers, loses information of small peaks in distribution)
def get_median_probabilities(probabilities_dict):
    median_probabilities_dict = {}
    
    for model_name in probabilities_dict:
        median_probabilities_dict[model_name] = {}

        for emotion in probabilities_dict[model_name]:
            probabilities = np.array(probabilities_dict[model_name][emotion])
            median_probabilities = np.median(probabilities, axis=0)
            median_probabilities_dict[model_name][emotion] = median_probabilities.tolist()

    return median_probabilities_dict


def get_distributions_2(model_name, data_path, output_path):
    activation_values_dict = get_activation_values(model_name, data_path)
    correlations_dict = correlation(activation_values_dict, output_path)
    
    # Get the average activations first
    avg_activations_dict = get_average_activations(activation_values_dict)
    
    # Then convert the average activations into probabilities
    probabilities_dict = activation_to_probabilities_2(avg_activations_dict)
    avg_probabilities_dict = get_average_probabilities(probabilities_dict)

    safe_file_path(activation_values_dict, output_path, file_name='activation_values.json')
    safe_file_path(probabilities_dict, output_path, file_name='probabilities.json')
    safe_file_path(avg_probabilities_dict, output_path, file_name='avg_probabilities.json')

    plot_average_probabilities(avg_probabilities_dict, output_path)

    return correlations_dict, avg_probabilities_dict, activation_values_dict


### Plotting functions
def get_all_plots(avg_probabilities_dict, correlation_dict, activation_values_dict, output_dir):

    plot_average_probabilities(avg_probabilities_dict, output_dir)
    #plot_correlation_indices(correlation_dict, output_dir)
    #plot_scatterplot_matrix(correlation_dict, output_dir)
    #plot_confusion_matrix(activation_values_dict, output_dir)
    #plot_correlation_matrix(activation_values_dict, output_dir)



def plot_average_probabilities(avg_probabilities_dict, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for model_name in avg_probabilities_dict:
        for emotion in sorted(avg_probabilities_dict[model_name]):
            avg_probabilities = avg_probabilities_dict[model_name][emotion]
            plt.figure()
            plt.plot(avg_probabilities)
            plt.title(f'Average probabilities for {emotion} in {model_name}')
            plt.xlabel('Image index')
            plt.ylabel('Average probability')
            # saving plots locally (.env file needs to be updated with path to plots folder)
            plt.savefig(os.path.join(output_dir, f'{model_name}_{emotion}.png'))
            plt.close()


def plot_correlation_indices(correlation_dict, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for model_name in correlation_dict:
        for label in sorted(correlation_dict[model_name]):
            correlation_matrix = np.array(correlation_dict[model_name][label])
            
            # replace NaN or infinite values with 0 (error in seaborn function)
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0, posinf=0.0, neginf=0.0)

            plt.figure(figsize=(10, 10))
            plt.imshow(correlation_matrix, cmap='hot', interpolation='nearest')
            plt.title(f'Correlation indices for {label} in {model_name}')
            plt.colorbar(label='Correlation')

            # add the correlation values to each cell
            for i in range(correlation_matrix.shape[0]):
                for j in range(correlation_matrix.shape[1]):
                    plt.text(j, i, format(correlation_matrix[i, j], '.3f'), 
                             ha='center', va='center', color='w')

            plt.savefig(os.path.join(output_dir, f'{model_name}_{label}_correlation.png'))
            plt.close()


def plot_confusion_matrix(activation_values_dict, output_dir, normalize=True):
    classes = sorted(list(next(iter(activation_values_dict.values())).keys()))
    y_true = []
    y_pred = []

    for model in activation_values_dict:
        for emotion in activation_values_dict[model]:
            for i, sublist in enumerate(activation_values_dict[model][emotion]):
                y_true.append(emotion)
                max_activation = max(sublist)
                predicted_class = emotion
                for class_name, values in activation_values_dict[model].items():
                    if class_name != emotion and i < len(values):
                        max_value = max(values[i])
                        if max_value > max_activation:
                            max_activation = max_value
                            predicted_class = class_name
                y_pred.append(predicted_class)

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(df_cm, annot=True, fmt=".3f" if normalize else "d", ax=ax, cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()


def plot_correlation_matrix(activation_values_dict, output_dir):
    classes = sorted(list(next(iter(activation_values_dict.values())).keys()))
    # following steps needed to have consistent lengths of lists to calculate correlation
    flattened = {class_name: [val for sublist in activation_values_dict[model][class_name] 
                              for val in sublist] 
                              for class_name in classes 
                              for model in activation_values_dict}
    
    min_length = min(len(lst) for lst in flattened.values())
    truncated = {class_name: lst[:min_length] for class_name, lst in flattened.items()}

    # calculate pairwise correlations
    corr = {class1: {class2: np.corrcoef(truncated[class1], truncated[class2])[0, 1] 
                     for class2 in classes} 
                     for class1 in classes}

    df = pd.DataFrame(corr)

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(df, annot=True, fmt=".3f", ax=ax, cmap='Blues')
    ax.set_title('Correlation Matrix')

    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()


def plot_scatterplot_matrix(correlation_dict, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for model_name in correlation_dict:
        for label in sorted(correlation_dict[model_name]):
            correlation_matrix = np.array(correlation_dict[model_name][label])
            
            # replace NaN or infinite values with 0 (error in seaborn function)
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            df = pd.DataFrame(correlation_matrix)

            # function for scatterplot matrix
            sns.pairplot(df)

            plt.title(f'Scatterplot matrix for {label} in {model_name}')
            plt.savefig(os.path.join(output_dir, f'{model_name}_{label}_scatterplot_matrix.png'))
            plt.close()


# Seaborn: concat all lists of each emtion into one list, plot them and calculate distrbution
# Plot correlation of emotions
            
# Correlation values get weighted by values of confusion matrix --> quantification of causality / relevance of correlation values for accuracy
# For correlations deemed relevant, should try to be set into context to coditioned true values -> if certain output nodes show high relevance these nodes should be taken into additional consideration
# after including the highest activation value
# since each outputnode can roughly be fit to a normal distribution with mean = highest activation value we could define a threshold 
# if activation is lower than a given threshold -> we use a comparison of distribution over all output nodes
# distribution of true values will therefore exclude activation values higher than the threshold
# would be interesting to know the avg activation value of the different output nodes given a wrong classification
# if the avg activation values in these cases match the combinations deemed relevant in the weighted correlation the prediction with distributions and KL-Divergence is possible
            

### Starting point: Input is activation_values_dict ###
            
def plot_activation_values(activation_values_dict, output_dir, plot_path):
    distributions = {}
    for model_name, model_data in activation_values_dict.items():
        distributions[model_name] = {}
        for emotion, values in model_data.items():

            df = pd.DataFrame(values, columns=['Value1', 'Value2', 'Value3', 'Value4', 'Value5', 'Value6'])
            df['Index'] = np.arange(len(df))
            
            # line plot of the activation values over their list indices
            for value_column in df.columns:
                if value_column == 'Index':
                    continue
                plt.figure(figsize=(10, 6))
                sns.lineplot(x='Index', y=value_column, data=df)
                plt.title(f'Activation Values for {emotion} in {model_name} ({value_column})')
                plt.savefig(f'{plot_path}/{model_name}_{emotion}_{value_column}_seaborn.png')
                plt.close()
            
            # distribution plot of the activation values
            for value_column in df.columns:
                if value_column == 'Index':
                    continue
                plt.figure(figsize=(10, 6))
                sns.distplot(df[value_column], bins=30)
                plt.title(f'Distribution of Activation Values for {emotion} in {model_name} ({value_column})')
                plt.savefig(f'{plot_path}/{model_name}_{emotion}_{value_column}_seaborn.png')
                plt.close()

            distributions[model_name][emotion] = df.drop(columns='Index').values.tolist()

    with open(f'{output_dir}/distributions_seaborn.json', 'w') as f:
        json.dump(distributions, f)


def plot_activation_values_2(activation_values_dict, output_dir, plot_path):
    activation_values_flat = {}
    distributions = {}
    for model_name, model_data in activation_values_dict.items():
        for emotion, values in model_data.items():

            if emotion not in activation_values_flat:
                activation_values_flat[emotion] = []
            
            df = pd.DataFrame(values, columns=['Value1', 'Value2', 'Value3', 'Value4', 'Value5', 'Value6'])
            
            for column in df.columns:
                activation_values_flat[emotion].extend(df[column].tolist())


    # combined plot of distribution and lineplot
    for emotion, values in activation_values_flat.items():
        df = pd.DataFrame(values, columns=['Values'])
        df['Index'] = np.arange(len(df))

        plt.figure(figsize=(10, 6))
        sns.lineplot(x='Index', y='Values', data=df)
        plt.title(f'Activation Values for {emotion}')
        plt.savefig(f'{plot_path}/{emotion}_seaborn.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.distplot(df['Values'], bins=6)
        plt.title(f'Distribution of Activation Values for {emotion}')
        plt.savefig(f'{plot_path}/{emotion}_seaborn.png')
        plt.close()

        # turning the flattened activationvalues into a distribution
        if emotion not in distributions:
            distributions[emotion] = []
        hist, _bin_edges = np.histogram(values, bins=6, density=True)
        distributions[emotion] = hist.tolist()


    with open(f'{output_dir}/distributions_seaborn.json', 'w') as f:
        json.dump(distributions, f)

    return distributions


def get_distributions_3(model_name, data_path, output_path, plot_path):

    activation_values_dict = get_activation_values(model_name, data_path)
    distributions = plot_activation_values_2(activation_values_dict, output_path, plot_path)

    return distributions


def calculate_kl_divergence(distributions, target_distributions):
    kl_divergences = []

    for i, target_distribution in enumerate(target_distributions):
        kl_divergence_row = {}

        for emotion, distribution in distributions.items():
            # calculate the KL divergence
            kl_divergence = entropy(target_distribution, distribution)
            kl_divergence_row[emotion] = kl_divergence

        kl_divergences.append(kl_divergence_row)

    # convert the list of dictionaries to a DataFrame
    kl_divergences_df = pd.DataFrame(kl_divergences)

    return kl_divergences_df


def get_inf_distributions(model_name, data_path):
    model_id, activation_values_df = apply_model(model_name, data_path)

    if not all(isinstance(label, str) for label in activation_values_df['file']):
        raise ValueError("The 'file' column should contain strings.")

    labels = activation_values_df['file'].values.tolist()

    # exclude the 'file' column 
    columns = [col for col in activation_values_df.columns if col != 'file']

    if not all(activation_values_df[col].dtype in ['int64', 'float64'] for col in columns):
        raise ValueError("The columns other than 'file' should contain numeric values.")

    distributions = {}

    for i, row in activation_values_df.iterrows():
        activation_values = row[columns].values.ravel()

        # replace NaN values with zero
        activation_values = np.nan_to_num(activation_values)

        hist, _bin_edges = np.histogram(activation_values, bins=6, density=True)
        
        # small constant to avoid division by zero
        epsilon = 1e-10
        hist = hist / (hist.sum() + epsilon)

        distributions[labels[i]] = hist.tolist()  # use the label as the key

    return distributions


def kl_inference(model_name, data_path):
    # load the distributions from 'distributions_seaborn.json'
    with open('activation_values/distributions_seaborn.json', 'r') as f:
        class_distributions = json.load(f)

    # get target distributions
    target_distributions = get_inf_distributions(model_name, data_path)

    # Print out the target_distributions dictionary for debugging
    print(target_distributions)

    kl_results = []

    # calculate the KL Divergences for each distribution in target_distributions
    for distribution_name, target_distribution in target_distributions.items():
        kl_divergences = calculate_kl_divergence(class_distributions, target_distribution)
        kl_divergences['distribution_name'] = distribution_name
        kl_results.append(kl_divergences)

    # convert the list of DataFrames to a single DataFrame
    kl_results_df = pd.concat(kl_results, ignore_index=True)

    return kl_results_df


def kl_confusion_matrix(kl_results_df: pd.DataFrame):
    """Return the confusion matrix."""
    LABEL_TO_STR = {i: str(i) for i in range(len(kl_results_df.columns) - 1)}
    LABEL_TO_STR[len(LABEL_TO_STR)] = 'inf'

    matrix = np.zeros((len(LABEL_TO_STR), len(LABEL_TO_STR)))

    for distribution_name, *kl_divergences in kl_results_df.values:
        label = label_from_path(distribution_name)
        if label is None:
            raise ValueError(f"Could not find label in distribution name {distribution_name}.")
        predictions = np.array(kl_divergences).argsort()[::-1]
        matrix[label, LABEL_TO_STR[predictions[0]]] += 1

    return pd.DataFrame(matrix, index=LABEL_TO_STR.values(), columns=LABEL_TO_STR.values())


def kl_divergence_accuracies(kl_divergence_results: pd.DataFrame, best=3) -> dict:
    """Return top1 and top3 accuracies based on KL divergence."""
    top_n = [0] * best

    for distribution_name, *kl_divergences in kl_divergence_results.values:
        label = label_from_path(distribution_name)
        if label is None:
            raise ValueError(f"Could not find label in distribution name {distribution_name}.")
        predictions = np.array(kl_divergences).argsort()[::-1]
        for i in range(len(top_n)):
            if label in predictions[:i + 1]:
                top_n[i] += 1
    return {f'top{i + 1}_acc': top_n[i] / len(kl_divergence_results) for i in range(len(top_n))}


### EXPERIMENTAL CODE ###
### USING PYTORCH DISTRIBUTION OBJECTS ###
import torch
from torch.distributions import Categorical
import torch
from torch.distributions import Categorical

bin_size = 6

def plot_activation_values_3(activation_values_dict, output_dir, plot_path):
    activation_values_flat = {}
    distributions = {}
    for model_name, model_data in activation_values_dict.items():
        for emotion, values in model_data.items():

            if emotion not in activation_values_flat:
                activation_values_flat[emotion] = []
            
            df = pd.DataFrame(values, columns=['Value1', 'Value2', 'Value3', 'Value4', 'Value5', 'Value6'])
            
            for column in df.columns:
                activation_values_flat[emotion].extend(df[column].tolist())

    # combined plot of distribution and lineplot
    for emotion, values in activation_values_flat.items():
        df = pd.DataFrame(values, columns=['Values'])
        df['Index'] = np.arange(len(df))

        plt.figure(figsize=(10, 6))
        sns.lineplot(x='Index', y='Values', data=df)
        plt.title(f'Activation Values for {emotion}')
        plt.savefig(f'{plot_path}/{emotion}_seaborn.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.distplot(df['Values'], bins=bin_size)
        plt.title(f'Distribution of Activation Values for {emotion}')
        plt.savefig(f'{plot_path}/{emotion}_seaborn.png')
        plt.close()

        # turning the flattened activationvalues into a distribution
        hist, _bin_edges = np.histogram(values, bins=bin_size, density=True)
        
        # Convert the histogram to a PyTorch tensor and create a Categorical distribution
        hist = torch.tensor(hist)
        hist = hist / hist.sum()  # Normalize the distribution
        hist_dist = Categorical(probs=hist)

        distributions[emotion] = hist.tolist()  # Convert the tensor back to a list

    with open(f'{output_dir}/distributions_pytorch.json', 'w') as f:
        json.dump(distributions, f)

    return distributions


def get_distributions_4(model_name, data_path, output_path, plot_path):
    activation_values_dict = get_activation_values(model_name, data_path)
    distributions = plot_activation_values_3(activation_values_dict, output_path, plot_path)

    # Convert the distributions to PyTorch distributions
    for emotion, distribution in distributions.items():
        distribution = torch.tensor(distribution)
        distributions[emotion] = Categorical(probs=distribution)

    return distributions


def get_inf_distributions_2(model_name, data_path):
    model_id, activation_values_df = apply_model(model_name, data_path)

    # Check that the 'file' column contains strings
    if not all(isinstance(label, str) for label in activation_values_df['file']):
        raise ValueError("The 'file' column should contain strings.")

    labels = activation_values_df['file'].values.tolist()

    # Exclude the 'file' column
    columns = [col for col in activation_values_df.columns if col != 'file']

    # Check that the other columns contain numeric values
    if not all(activation_values_df[col].dtype in ['int64', 'float64'] for col in columns):
        raise ValueError("The columns other than 'file' should contain numeric values.")

    distributions = {}

    # Apply the histogram calculation to each row in the DataFrame
    for i, row in activation_values_df.iterrows():
        activation_values = row[columns].values.ravel()

        # Replace NaN values with zero
        activation_values = np.nan_to_num(activation_values)

        hist, _bin_edges = np.histogram(activation_values, bins=bin_size, density=True)
        
        # Add a small constant to avoid division by zero
        epsilon = 1e-10
        hist = hist / (hist.sum() + epsilon)

        # Check if the sum of the distribution is zero
        if hist.sum() == 0:
            print(f"Warning: The sum of the distribution for label {labels[i]} is zero.")

        # Convert the histogram to a PyTorch tensor and create a Categorical distribution
        hist = torch.tensor(hist)
        hist = Categorical(probs=hist)

        distributions[labels[i]] = hist  # use the label as the key

    return distributions


def calculate_kl_divergence_2(distributions, target_distribution):
    kl_divergence_row = {}

    for emotion, distribution in distributions.items():
        # calculate the KL divergence
        kl_divergence = torch.distributions.kl.kl_divergence(target_distribution, distribution)
        kl_divergence_row[emotion] = kl_divergence.item()

    return kl_divergence_row


def kl_inference_2(model_name, data_path):
    # load the distributions from 'distributions_seaborn.json'
    with open('activation_values/distributions_pytorch.json', 'r') as f:
        class_distributions = json.load(f)

    # Convert the class_distributions to PyTorch distributions
    for emotion, distribution in class_distributions.items():
        distribution = torch.tensor(distribution)
        distribution = distribution / distribution.sum()  # Normalize the distribution
        class_distributions[emotion] = Categorical(probs=distribution)

    # get target distributions
    target_distributions = get_inf_distributions_2(model_name, data_path)

    # Print out the target_distributions dictionary for debugging
    #print(target_distributions)

    kl_results = []
    labels = []

    # calculate the KL Divergences for each distribution in target_distributions
    for distribution_name, target_distribution in target_distributions.items():
        kl_divergences = calculate_kl_divergence_2(class_distributions, target_distribution)
        kl_divergences['distribution_name'] = distribution_name
        kl_results.append(kl_divergences)
        labels.append(distribution_name.split('_')[-1])  # assuming the label is the last part of the distribution_name

    # convert the list of dictionaries to a DataFrame
    kl_results_df = pd.DataFrame(kl_results)

    return kl_results_df, labels


def calculate_top_n_accuracies_kl(kl_results_df, labels):
    top1 = 0.0
    top3 = 0.0
    top1_labels = []  # List to store the top predicted label for each true label

    for i, row in kl_results_df.iterrows():
        # Get the true label and remove the '.jpg' extension
        true_label = labels[i].replace('.jpg', '')

        # Get the predicted labels sorted by KL divergence (ascending)
        pred_labels = row.drop('distribution_name').sort_values().index.tolist()

        # Convert the predicted labels from encoded labels to class names
        pred_labels = [LABEL_TO_STR[int(label)] for label in pred_labels]

        # Store the top predicted label
        top1_labels.append(pred_labels[0])

        # Check if the true label is in the top 3
        if true_label in pred_labels[:3]:
            top3 += 1
            if true_label == pred_labels[0]:
                top1 += 1

    top1 /= len(kl_results_df)
    top3 /= len(kl_results_df)
    return top1, top3, top1_labels


def generate_confusion_matrix(true_labels, top1_labels):
    # Remove the '.jpg' extension from the true labels
    true_labels = [label.replace('.jpg', '') for label in true_labels]

    # Generate the confusion matrix
    cm = confusion_matrix(true_labels, top1_labels)

    return cm


def get_unique_distributions(distributions, plot_path):
    unique_distributions = {}
    seen_labels = set()

    for label, distribution in distributions.items():
        # Get the label without the file extension
        label_without_extension = label.replace('.jpg', '')

        if label_without_extension not in seen_labels:
            unique_distributions[label] = distribution
            seen_labels.add(label_without_extension)

            # Plot the distribution
            values = distribution.probs.tolist()  # Get the probabilities from the distribution
            df = pd.DataFrame(values, columns=['Values'])
            df['Index'] = np.arange(len(df))

            plt.figure(figsize=(10, 6))
            sns.lineplot(x='Index', y='Values', data=df)
            plt.title(f'Distribution for {label_without_extension}')
            plt.savefig(f'{plot_path}/{label_without_extension}_seaborn.png')
            plt.close()

    return unique_distributions
    