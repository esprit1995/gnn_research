import os
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from termcolor import cprint

from datasets import IMDB_ACM_DBLP_from_GTN, ACM_HAN
from statistics import mean


def evaluate_clustering(node_embeddings: torch.tensor, ids: np.array, labels: np.array):
    """
    evaluate performance of node clustering on given embeddings using k-means algorithm
    :param node_embeddings: node embeddings in torch.tensor format, shape = [n_nodes, n_features]
    :param ids: node ids of the test set to evaluate on
    :param labels:  corresponding labels
    :return:
    """
    embs = node_embeddings.detach()[ids]
    return kmeans_node_clustering(embs, labels)


def kmeans_node_clustering(node_embeddings: torch.tensor,labels: np.array, runs: int = 10) -> tuple:
    """
    perform k-means clustering on given node embeddings and evaluate it using NMI and ARI scores
    number of clusters is inferred from the number of different labels
    :param node_embeddings: node embeddings in torch.tensor format, shape = [n_nodes, n_features]
    :param true_labels: true labels of the node embeddings, shape = [n_nodes,]
    :param runs: since kmeans is dependent on the center initialization, algo is rerun __runs__ times
    :return: tuple(NMI, ARI)
    """
    emb = node_embeddings.numpy()
    n_clusters = np.unique(labels).size

    # fix random state for reproducible results
    random_states = list(range(runs))
    nmis = list()
    aris = list()
    for rs in random_states:
        kmeans_labels = KMeans(n_clusters=n_clusters, random_state=rs).fit(emb).labels_
        NMI = normalized_mutual_info_score(kmeans_labels, labels)
        ARI = adjusted_rand_score(kmeans_labels, labels)
        nmis.append(NMI)
        aris.append(ARI)

    print()
    cprint("--------------------------", color='blue')
    cprint("Co-Clustering task, NMI:  " + str(mean(nmis)), color='blue')
    cprint("Co-Clustering task, ARI:  " + str(mean(aris)), color='blue')
    cprint("--------------------------", color='blue')
    print()
    return mean(nmis), mean(aris)
