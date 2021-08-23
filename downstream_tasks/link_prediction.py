import torch
import numpy as np
import warnings

from utils.tools import heterogeneous_negative_sampling_naive
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

from termcolor import cprint


def evaluate_link_prediction(node_embeddings: torch.tensor,
                             node_type_mask: torch.tensor,
                             edge_index_dict: dict,
                             verbose: bool = False):
    link_data = np.array([[], [], []]).reshape((-1, 3))
    for key in list(edge_index_dict.keys()):
        central, positive, negative = heterogeneous_negative_sampling_naive(edge_index_dict[key], node_type_mask)
        to_append_positive = np.stack([central.numpy().T,
                                       positive.numpy().T,
                                       np.array([1] * central.shape[0]).T], axis=-1)
        to_append_negative = np.stack([central.numpy().T,
                                       negative.numpy().T,
                                       np.array([0] * central.shape[0]).T], axis=-1)
        to_append = np.vstack([to_append_negative, to_append_positive])
        link_data = np.vstack([link_data, to_append])

    train_edges, test_edges = train_test_split(link_data,
                                               test_size=0.7,
                                               random_state=69,
                                               shuffle=True,
                                               stratify=link_data[:, 2])
    roc_auc_macro, f1_macro = logreg_link_prediction(train_edges, test_edges,
                                                     node_embeddings=node_embeddings,
                                                     verbose=verbose)
    return roc_auc_macro, f1_macro


def logreg_link_prediction(train_data: np.array, test_data: np.array,
                           node_embeddings: torch.tensor,
                           verbose: bool = False):
    warnings.simplefilter('ignore')
    emb_dimension = node_embeddings.shape[1]
    train = node_embeddings.detach().numpy()[train_data[:, :2].astype(int)].reshape((-1, 2 * emb_dimension))
    train_labels = train_data[:, 2].astype(int)
    test = node_embeddings.detach().numpy()[test_data[:, :2].astype(int)].reshape((-1, 2 * emb_dimension))
    test_labels = test_data[:, 2].astype(int)

    classifier = LogisticRegression()
    classifier.fit(train, train_labels)
    test_predictions = classifier.predict(test)

    roc_auc = roc_auc_score(test_labels, test_predictions, average='macro', multi_class='ovo')
    f1 = f1_score(test_labels, test_predictions, average='macro')
    if verbose:
        print()
        cprint("--------------------------", color='blue')
        cprint("Link prediction task, roc_auc (macro average):  " + str(roc_auc), color='blue')
        cprint("Link prediction task, f1 (macro average):  " + str(f1), color='blue')
        cprint("--------------------------", color='blue')
        print()
        warnings.simplefilter('default')
    return roc_auc, f1
