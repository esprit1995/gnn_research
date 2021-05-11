import numpy as np
import torch
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from termcolor import cprint


def evaluate_classification(node_embeddings: torch.tensor,
                            ids_train: np.array, labels_train: np.array,
                            ids_test: np.array, labels_test: np.array,
                            verbose: bool = True):
    """
    evaluate performance of node classification on given embeddings using logistic regression
    :param ids_train: node ids to train on
    :param labels_train: corresponding labels
    :param ids_test: node ids to test on
    :param labels_test: corresponding labels (ground truth)
    :param node_embeddings: node embeddings in torch.tensor format, shape = [n_nodes, n_features]
    :param verbose: whether to print the eval metrics in the console
    :return:
    """

    embs_train = node_embeddings.detach()[ids_train]
    embs_test = node_embeddings.detach()[ids_test]
    return logreg_node_classification(embs_train, labels_train, embs_test, labels_test, verbose=verbose)


def logreg_node_classification(train_embeddings: torch.tensor, labels_train: np.array,
                               test_embeddings: torch.tensor, labels_test: np.array,
                               verbose: bool = True) -> tuple:
    """
    perform logreg classification on given node embeddings and evaluate it using Macro- and Micro- F1 scores
    :param train_embeddings: train node embeddings in torch.tensor format, shape = [n_nodes_train, n_features]
    :param labels_train: true labels of the train node embeddings, shape = [n_nodes_train,]
    :param test_embeddings: test node embeddings in torch.tensor format, shape = [n_nodes_test, n_features]
    :param labels_test: true labels of the test node embeddings, shape = [n_nodes_test,]
    :param verbose: whether to print the eval metrics in the console
    :return: tuple(microF1, macroF1)
    """
    warnings.simplefilter('ignore')
    train = train_embeddings.numpy()
    test = test_embeddings.numpy()

    classifier = LogisticRegression()
    classifier.fit(train, labels_train)
    test_predictions = classifier.predict(test)

    microF1, macroF1 = f1_score(labels_test, test_predictions, average='micro'), f1_score(labels_test, test_predictions,
                                                                                          average='macro')
    if verbose:
        print()
        cprint("--------------------------", color='blue')
        cprint("Co-Classification task, microF1:  " + str(microF1), color='blue')
        cprint("Co-Classification task, macroF1:  " + str(macroF1), color='blue')
        cprint("--------------------------", color='blue')
        print()
        warnings.simplefilter('default')
    return microF1, macroF1
