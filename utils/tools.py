import datetime

import torch
import numpy as np
import multiprocessing as mp
from torch_geometric.typing import Adj

from sklearn.preprocessing import OneHotEncoder

from typing import Dict, Tuple, Any


def heterogeneous_negative_sampling_naive(edge_index: Adj,
                                          node_idx_type: torch.Tensor) -> tuple:
    """
    alteration of torch_geometric.utils.structured_negative_sampling to accommodate for
    heterogeneous graphs
    :param edge_index: edge index of the graph/batch
    :param node_idx_type: tensor, node_id_type[i] = type of the node with id = i
    :return: tensor [3xn_edges] containing node (central, positive, negative) triplets
    """
    positive_node_types = list(set([elem.item() for elem in node_idx_type[edge_index[1]]]))
    negative_nodes = torch.ones(edge_index.shape[1]) * (-1)
    for pos_node_type in positive_node_types:
        # consider only edges that terminate in a given node type
        sub_edge_index = edge_index[:, np.where(node_idx_type[edge_index[1].numpy()] == pos_node_type)[0]]
        # list of all possible node ids of that type
        allowed_neg_idx = np.where(node_idx_type.numpy() == pos_node_type)[0]

        i, j = sub_edge_index.to('cpu')
        idx_1 = i * allowed_neg_idx.max() + j

        k = torch.randint(allowed_neg_idx.size, (i.size(0),), dtype=torch.long)
        idx_2 = i * allowed_neg_idx.max() + allowed_neg_idx[k]
        mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)
        rest = mask.nonzero(as_tuple=False).view(-1)
        while rest.numel() > 0:  # pragma: no cover
            tmp = torch.randint(allowed_neg_idx.size, (rest.numel(),), dtype=torch.long)
            idx_2 = i[rest] * allowed_neg_idx.max() + allowed_neg_idx[tmp]
            mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)
            k[rest] = tmp
            rest = rest[mask.nonzero(as_tuple=False).view(-1)]
        negative_nodes[np.where(node_idx_type[edge_index[1].numpy()] == pos_node_type)[0]] = torch.tensor(
            allowed_neg_idx[k], dtype=torch.float)

    return edge_index[0].long(), edge_index[1].long(), negative_nodes.long()


def node_type_encoding(node_features: np.array, node_type_mask: np.array):
    """
    1-hot encode node types and add them to node_features
    :param node_features: [n_nodes x n_features] feature matrix
    :param node_type_mask: [n_nodes] array. array[i] = node_type(i)
    :return: updates feature matrix
    """
    if node_features.shape[0] != node_type_mask.shape[0]:
        raise ValueError('node_type_encoding(): node features and node_type_mask have incompatible shapes')
    encoder = OneHotEncoder(sparse=False)
    type_feats = np.asarray(encoder.fit_transform(node_type_mask.reshape(-1, 1)))
    return torch.tensor(np.concatenate((node_features, type_feats), axis=1))


# ############################################################
# ####### Graphlet sampling: metapath-like graphlets #########
# ############################################################


def edge_index_to_adj_dict(edge_index: Dict, node_type_mask: torch.tensor, between_types: Tuple) -> np.array:
    """
    creates a numpy adjacency matrix between the specified node types based on the passed edge_index
    :param edge_index: a dict containing edges. edge_index[(type_1, type_2)] contains edges between
                       nodes of types type_1, type_2
    :param node_type_mask: numpy array containing node types, node_types[i] = type of node with index i.
    :param between_types: tuple (type_1, type_2); should be on of the keys of edge_index
    :return: adjacency dictionary between (type_1, type_2) in numpy format
    """
    if between_types not in list(edge_index.keys()):
        raise ValueError("edge_index_to_adj_dict(): type pair not found in the edge index")
    typed_edge_index = edge_index[between_types].numpy()
    head_type_range = np.sort(np.where(node_type_mask == int(between_types[0]))[0])
    adj_dict = dict()
    for head_type_idx in head_type_range:
        if head_type_idx in typed_edge_index[0]:
            adj_dict[str(head_type_idx)] = typed_edge_index[1][np.where(typed_edge_index[0] == head_type_idx)]
        else:
            adj_dict[str(head_type_idx)] = np.array([])
    return adj_dict


def make_step(current: int, adj_dict) -> int:
    try:
        if adj_dict[str(current)].size > 0:
            return np.random.choice(adj_dict[str(current)])
        else:
            return -1
    except Exception as e:
        raise e


def sample_instance(metapath_, adj_dicts_, starting_points_, random_seed) -> tuple:
    metapath_instance = list()
    np.random.seed(random_seed)
    metapath_current = np.random.choice(starting_points_)
    metapath_instance.append(metapath_current)
    for nstep in range(1, len(metapath_)):
        edge_type = (metapath_[nstep - 1], metapath_[nstep])
        try:
            metapath_current = make_step(metapath_current, adj_dicts_[edge_type])
        except Exception as e:
            print('Sampling process failed due to exception: ' + str(e))
            raise e
        if metapath_current == -1:
            return None
        metapath_instance.append(metapath_current)
    return tuple(metapath_instance)


def sample_metapath_instances(metapath: Tuple, n: int, pyg_graph_info: Any, nworkers: int = 4,
                              parallel: bool = False) -> list:
    """
    sample a predefined number of metapath instances in a given graph
    :param metapath: a tuple describing a metapath. For instance, ('0', '1', '2', '1', '0')
    :param n: how many instances to randomly sample from a graph
    :param pyg_graph_info: variable containing at least 'edge_index_dictionary', 'node_type_mask'
    :param nworkers: parallel processing param
    :param parallel: whether to use parallel processing
    :return: a list of sampled metapath instances
    """
    np.random.seed(69)
    # get all possible starting points for the given metapath template
    starting_points = np.where(pyg_graph_info['node_type_mask'].numpy() == int(metapath[0]))[0]

    # get adjacency dictionaries needed for the given metapath template
    adj_dicts = dict()
    for nedge in range(1, len(metapath)):
        adj_dicts[(metapath[nedge - 1], metapath[nedge])] = edge_index_to_adj_dict(pyg_graph_info['edge_index_dict'],
                                                                                   pyg_graph_info['node_type_mask'],
                                                                                   (metapath[nedge - 1],
                                                                                    metapath[nedge]))

    # instance sampling
    if parallel:
        pool = mp.Pool(processes=nworkers)
        results = [pool.apply_async(sample_instance, args=(metapath, adj_dicts, starting_points, i)) for i in range(n)]
        mp_instances = [p.get() for p in results]
        return mp_instances
    else:
        results = list()
        for i in range(n):
            instance = sample_instance(metapath, adj_dicts, starting_points, i)
            if instance is not None:
                results.append(sample_instance(metapath, adj_dicts, starting_points, i))
        return results


# ############################################################
# ############################################################
# ############################################################


class EarlyStopping(object):
    """
    credits: DGL oficial github examples at
    https://github.com/dmlc/dgl/blob/master/examples/pytorch/han/utils.py
    """

    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))
