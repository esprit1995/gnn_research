import os
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from termcolor import cprint

from datasets import IMDB_ACM_DBLP, ACM_HAN
from statistics import mean


def evaluate_clustering(args, node_embeddings: torch.tensor):
    """
    evaluate performance of node clustering on given embeddings using k-means algorithm
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

    ids = dataset['test_id_label'][0].numpy()
    labels = dataset['test_id_label'][1]

    embs = node_embeddings.detach()[ids]
    return kmeans_node_clustering(embs, labels)


def kmeans_node_clustering(node_embeddings: torch.tensor, true_labels: torch.tensor, runs: int = 10) -> tuple:
    """
    perform k-means clustering on given node embeddings and evaluate it using NMI and ARI scores
    number of clusters is inferred from the number of different labels
    :param node_embeddings: node embeddings in torch.tensor format, shape = [n_nodes, n_features]
    :param true_labels: true labels of the node embeddings, shape = [n_nodes,]
    :param runs: since kmeans is dependent on the center initialization, algo is rerun __runs__ times
    :return: tuple(NMI, ARI)
    """
    emb = node_embeddings.numpy()
    labels = true_labels.numpy()
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
    cprint("Clustering task, NMI:  " + str(mean(nmis)), color='blue')
    cprint("Clustering task, ARI:  " + str(mean(aris)), color='blue')
    cprint("--------------------------", color='blue')
    print()
    return mean(nmis), mean(aris)
