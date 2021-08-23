import numpy as np
import torch
from downstream_tasks.node_classification import evaluate_classification
from downstream_tasks.node_clustering import evaluate_clustering
from downstream_tasks.link_prediction import evaluate_link_prediction


def evaluate_clu_cla_GTN_NSHE_datasets(dataset, embeddings, verbose: bool = True):
    id_label_clustering = np.vstack([dataset['train_id_label'],
                                     dataset['test_id_label'],
                                     dataset['valid_id_label']])
    id_label_classification_train = np.vstack([dataset['train_id_label'],
                                               dataset['valid_id_label']])
    id_label_classification_test = dataset['test_id_label']
    # --> clustering, NMI, ARI metrics of K-means
    if type(embeddings) == type(np.array([0, 0])):
        embeddings = torch.tensor(embeddings)
    NMI, ARI = evaluate_clustering(embeddings, ids=id_label_clustering[:, 0],
                                   labels=id_label_clustering[:, 1], verbose=verbose)
    # --> classification, microF1, macroF1 metrics of logreg
    microF1, macroF1 = evaluate_classification(embeddings,
                                               ids_train=id_label_classification_train[:, 0],
                                               labels_train=id_label_classification_train[:, 1],
                                               ids_test=id_label_classification_test[:, 0],
                                               labels_test=id_label_classification_test[:, 1],
                                               verbose=verbose)
    return NMI, ARI, microF1, macroF1


def evaluate_link_prediction_GTN_NSHE_datasets(dataset, embeddings, verbose: bool = True):
    if type(embeddings) == type(np.array([0, 0])):
        embeddings = torch.tensor(embeddings)
    roc_auc, f1 = evaluate_link_prediction(node_embeddings=embeddings,
                                           node_type_mask=dataset['node_type_mask'],
                                           edge_index_dict=dataset['edge_index_dict'],
                                           verbose=verbose)
    return roc_auc, f1
