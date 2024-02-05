import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from utils import LABEL_TO_STR

"""
very hard pictures:
- img326_happiness -> none
- img232_disgust -> none
- img460_sadness -> hand in face
- img563_anger -> neutral
interesting pictures:
- img048_suprise -> sadness
- img190_fear -> surprise
- img240_disgust -> anger
- img372_happiness -> surprise
- img377_happiness -> disgust
- img585_anger -> fear
"""


def pca_graph(model_id, inference_results: pd.DataFrame, softmax=False):
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
