import os

import captum
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch import nn

from utils import LABEL_TO_STR, get_images_and_labels

import warnings


def pca_graph(model_id, inference_results: pd.DataFrame, softmax=False):
    """Create pca graph of model output."""
    # apply softmax on rows without the file name
    if softmax:
        inference_results.iloc[:, 1:] = inference_results.iloc[:, 1:].apply(lambda x: np.exp(x) / np.sum(np.exp(x)),
                                                                            axis=1)
    # extract the top 2 principal components
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(inference_results.values[:, 1:])
    # create a dataframe with the principal components
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    # get label: take highest value for file np.argmax
    label = [LABEL_TO_STR[x] for x in inference_results.values[:, 1:].argmax(axis=1)]

    true_label = []
    for file in inference_results.values[:, 0]:
        for emotion in LABEL_TO_STR.values():
            if emotion in file:
                true_label.append(emotion)

    principal_df['label'] = true_label
    # print the dataframe
    print(principal_df.head())
    # plot the graph
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.grid()
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title(f"2 component PCA ({model_id})", fontsize=20)
    targets = list(LABEL_TO_STR.values())
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for target, color in zip(targets, colors):
        indices_to_keep = principal_df['label'] == target
        ax.scatter(principal_df.loc[indices_to_keep, 'PC1'],
                   principal_df.loc[indices_to_keep, 'PC2'],
                   c=color,
                   s=50)
    ax.legend(targets)
    plt.show()


def explain_all(model: nn.Module, data_path, path_contains=None, save_path=None, window_size=(8, 8)):
    """Use all methods to explain the model. Display in matplotlib grid."""
    warnings.filterwarnings("ignore")
    tensors, labels, paths = get_images_and_labels(data_path, limit=1, path_contains=path_contains)
    fig, ax = plt.subplots(3, 3, figsize=(8, 8))
    tensor, label, path = tensors[0], labels[0], paths[0]
    ax = ax.reshape(-1)
    results = {}
    for i, (method_name, method) in enumerate(METHODS.items()):
        tensor.requires_grad = True
        if method_name == 'occlusion':
            result = method(model, tensor.unsqueeze(0), label, window_size=window_size)
        else:
            result = method(model, tensor.unsqueeze(0), label)
        ax[i].axis('off')
        ax[i].imshow(result, cmap='gray')
        results[method_name] = result
        ax[i].title.set_text(f"{method_name}")

    if save_path:
        base_path = f"{save_path}/{path.split('/')[-1].split('.')[0]}"
        os.makedirs(base_path, exist_ok=True)
        for method_name, result in results.items():
            full_path = os.path.join(base_path, path.split('/')[-1].split('.')[0] + f"_{method_name}" + "_2.jpg")
            print(f"Saving result to {full_path}")
            cv2.imwrite(full_path, np.uint8(result * 255.))
    else:
        plt.show()


def explain_with_method(model: nn.Module, method, data_path, examples=10, random=True,
                        path_contains=None, save_path=None, only_return_first=False, window_size=(8, 8)):
    """Generates visual explanation of the chosen model with the chosen method."""
    warnings.filterwarnings("ignore")
    if method not in METHODS:
        raise ValueError(f"Method not supported: {method}, supported methods: {list(METHODS.keys())}")
    tensors, labels, paths = get_images_and_labels(data_path, limit=examples,
                                                   random=random, path_contains=path_contains)

    for tensor, l, p in zip(tensors, labels, paths):
        tensor.requires_grad = True
        prediction = model(tensor.unsqueeze(0)).argmax().item()
        if method == 'occlusion':
            result = METHODS[method](model, tensor.unsqueeze(0), l, window_size=window_size)
        else:
            result = METHODS[method](model, tensor.unsqueeze(0), l)
        if only_return_first:
            return result
        fig, ax = plt.subplots(1, 2)
        ax[0].axis('off')
        ax[0].imshow(tensor.permute(1, 2, 0))
        ax[0].title.set_text(f"{p}")

        ax[1].axis('off')
        ax[1].imshow(result, cmap='gray')
        ax[1].title.set_text(f"Prediction {LABEL_TO_STR[prediction]}")
        if save_path:
            path = f"{save_path}/{p.split('/')[-1]}_{method}.png"
            print(f"Saving result to {path}")
            # save result to file
            cv2.imwrite(path, np.uint8(result * 255.))
        else:
            plt.show()


def occlusion(model, tens, label, window_size=(8, 8)):
    occ_model = captum.attr.Occlusion(model)
    result = occ_model.attribute(tens, target=label, sliding_window_shapes=(3, *window_size))
    return stardardize(result)


def guided_backprop(model, tens, label):
    guided_bp = captum.attr.GuidedBackprop(model)
    result = guided_bp.attribute(tens, target=label)
    return stardardize(result)


def gradcam(model, tens, label):
    gradcam_model = captum.attr.LayerGradCam(model, model.conv_block3)
    result = gradcam_model.attribute(tens, target=label)
    result = result.repeat(1, 3, 1, 1)
    return stardardize(result)


def guided_gradcam(model, tens, label):
    guided_gc_model = captum.attr.GuidedGradCam(model, model.conv_block3)
    result = guided_gc_model.attribute(tens, target=label)
    return stardardize(result)


def deconv(model, tens, label):
    deconv_model = captum.attr.Deconvolution(model)
    result = deconv_model.attribute(tens, target=label)
    return stardardize(result)


def saliency_map(model, tens, label):
    saliency_model = captum.attr.Saliency(model)
    result = saliency_model.attribute(tens, target=label)
    return stardardize(result)


def deep_lift(model, tens, label):
    deep_lift_model = captum.attr.DeepLift(model)
    result = deep_lift_model.attribute(tens, target=label)
    return stardardize(result)


def input_x_gradient(model, tens, label):
    ig = captum.attr.InputXGradient(model)
    result = ig.attribute(tens, target=label)
    return stardardize(result)


def pertube(model, tens, label):
    old_pred = model(tens).argmax().item()
    print(f"Old prediction: {LABEL_TO_STR[old_pred]}")
    pertube_model = captum.robust.FGSM(model)
    result = pertube_model.perturb(tens, 0.01, label)
    new_pred = model(result).argmax().item()
    print(f"New prediction: {LABEL_TO_STR[new_pred]}")
    return stardardize(result)


def stardardize(tens):
    result = tens.squeeze().permute(1, 2, 0).detach().numpy()
    result -= result.min()
    result /= result.max()
    return result


METHODS = {
    'gradcam': gradcam,
    'guided-gradcam': guided_gradcam,
    'guided-backprop': guided_backprop,
    'deconv': deconv,
    'occlusion': occlusion,
    'saliency': saliency_map,
    'deep-lift': deep_lift,
    'input-x-gradient': input_x_gradient,
    'pertube': pertube,
}
