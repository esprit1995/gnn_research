import datetime

import torch
import numpy as np
from torch_geometric.typing import Adj

from sklearn.preprocessing import OneHotEncoder


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
