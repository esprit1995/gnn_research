import pandas as pd
import numpy as np
import os
import torch
from downstream_tasks.node_clustering import evaluate_clustering
from downstream_tasks.node_classification import evaluate_classification
from datasets import DBLP_ACM_IMDB_from_NSHE, IMDB_ACM_DBLP_from_GTN
from data_transforms import NSHE_for_rgcn, GTN_for_rgcn

DEFAULT_COMP_EMB_PATH = '/home/ubuntu/msandal_code/PyG_playground/competitors_perf/competitor_embeddings'
DEFAULT_DATA_PATH = '/home/ubuntu/msandal_code/PyG_playground/data/'


def evaluate_competitor(path_to_embs: str = DEFAULT_COMP_EMB_PATH,
                        dataset: str = 'dblp',
                        from_paper: str = 'nshe',
                        evaluate_architecture: str='nshe'):
    """
    evaluate clustering/classification on a chosen dataset using
    embeddings produced by DeepWalk
    :param path_to_embs: where the embeddings are stored
    :param dataset: which dataset to evaluate. Can be: ['dblp', 'imdb', 'acm']
    :param from_paper: from which paper the dataset version comes. Can be: ['nshe', 'gtn']
    :param evaluate_architecture: which architecture to evaluate. Possible values: ['deepwalk', 'nshe']
    :return:
    """
    dataset = str(dataset).lower()
    from_paper = str(from_paper).lower()
    evaluate_architecture = str(evaluate_architecture).lower()
    assert dataset in ['dblp', 'acm', 'imdb'], \
        'evaluate_competitor(): invalid dataset requested'
    assert from_paper in ['nshe', 'gtn'], \
        'evaluate_competitor(): invalid from_paper argument'
    assert evaluate_architecture in ['nshe', 'deepwalk'], \
        'evaluate_competitor(): cannot evaluate given architecture: ' + str(evaluate_architecture)
    # === get embeddings
    if evaluate_architecture == 'deepwalk':
        emb_filename = dataset + "_from_" + from_paper + "_deepwalk.embeddings"
        with open(os.path.join(path_to_embs, emb_filename)) as f:
            lines = (line for line in f)
            embs = np.loadtxt(lines, delimiter=' ', skiprows=1)
        embs = embs[np.argsort(embs[:, 0])]  # sort by node id, which is the first column
        embs = embs[:, 1:]  # remove the first column, as it is not a feature
    elif evaluate_architecture == 'nshe':
        emb_filename = dataset + "_from_" + from_paper + "_nshe_embeddings.npy"
        embs = np.load(os.path.join(path_to_embs, emb_filename))

    if from_paper == 'nshe':
        dataset = DBLP_ACM_IMDB_from_NSHE(root=os.path.join(DEFAULT_DATA_PATH, 'NSHE'), name=dataset.lower())[0]
    elif from_paper == 'gtn':
        dataset = IMDB_ACM_DBLP_from_GTN(root=os.path.join(DEFAULT_DATA_PATH, 'IMDB_ACM_DBLP', dataset.upper()),
                                         name=dataset.upper())[0]
        dataset['train_id_label'] = dataset['train_id_label'].numpy().T
        dataset['test_id_label'] = dataset['test_id_label'].numpy().T
        dataset['valid_id_label'] = dataset['valid_id_label'].numpy().T
    id_label_clustering = np.vstack([dataset['train_id_label'],
                                     dataset['test_id_label'],
                                     dataset['valid_id_label']])
    id_label_classification_train = np.vstack([dataset['train_id_label'],
                                               dataset['valid_id_label']])
    id_label_classification_test  = dataset['test_id_label']
    # --> clustering, NMI, ARI metrics of K-means
    NMI, ARI = evaluate_clustering(torch.tensor(embs), ids=id_label_clustering[:, 0],
                                   labels=id_label_clustering[:, 1])
    # --> classification, microF1, macroF1 metrics of logreg
    microF1, macroF1 = evaluate_classification(torch.tensor(embs),
                                               ids_train=id_label_classification_train[:, 0],
                                               labels_train=id_label_classification_train[:, 1],
                                               ids_test=id_label_classification_test[:, 0],
                                               labels_test=id_label_classification_test[:, 1])
