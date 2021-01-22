import os

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from termcolor import cprint

from datasets import IMDB_ACM_DBLP, ACM_HAN


def evaluate_classification(args, node_embeddings: torch.tensor):
    """
    evaluate performance of node classification on given embeddings using logistic regression
    :param args: experiment argparse arguments
    :param node_embeddings: node embeddings in torch.tensor format, shape = [n_nodes, n_features]
    :return:
    """
    if args.dataset in ['IMDB', 'DBLP', 'ACM']:
        datadir = '/home/ubuntu/msandal_code/PyG_playground/data/IMDB_ACM_DBLP'
        dataset = IMDB_ACM_DBLP(root=os.path.join(datadir, args.dataset), name=args.dataset)[0]
    elif args.dataset == 'ACM_HAN':
        datadir = '/home/ubuntu/msandal_code/PyG_playground/data/ACM_HAN'
        dataset = ACM_HAN(root=datadir)[0]
    else:
        raise NotImplementedError('evaluate_clustering: requested dataset unknown')

    ids_train = dataset['train_id_label'][0].numpy()
    labels_train = dataset['train_id_label'][1]

    ids_test = dataset['test_id_label'][0].numpy()
    labels_test = dataset['test_id_label'][1]

    embs_train = node_embeddings.detach()[ids_train]
    embs_test = node_embeddings.detach()[ids_test]
    return logreg_node_classification(embs_train, labels_train, embs_test, labels_test)


def logreg_node_classification(train_embeddings: torch.tensor, train_labels: torch.tensor,
                               test_embeddings: torch.tensor, test_labels: torch.tensor) -> tuple:
    """
    perform logreg classification on given node embeddings and evaluate it using Macro- and Micro- F1 scores
    :param train_embeddings: train node embeddings in torch.tensor format, shape = [n_nodes_train, n_features]
    :param train_labels: true labels of the train node embeddings, shape = [n_nodes_train,]
    :param test_embeddings: test node embeddings in torch.tensor format, shape = [n_nodes_test, n_features]
    :param test_labels: true labels of the test node embeddings, shape = [n_nodes_test,]
    :return: tuple(microF1, macroF1)
    """
    train = train_embeddings.numpy()
    labels_train = train_labels.numpy()
    test = test_embeddings.numpy()
    labels_test = test_labels.numpy()

    classifier = LogisticRegression()
    classifier.fit(train, labels_train)
    test_predictions = classifier.predict(test)

    microF1, macroF1 = f1_score(labels_test, test_predictions, average='micro'), f1_score(labels_test, test_predictions, average='macro')

    print()
    cprint("--------------------------", color='blue')
    cprint("Classification task, microF1:  " + str(microF1), color='blue')
    cprint("Classification task, macroF1:  " + str(macroF1), color='blue')
    cprint("--------------------------", color='blue')
    print()
    return microF1, macroF1
