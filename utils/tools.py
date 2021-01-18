import torch
import numpy as np
import torch_geometric as tg
from torch_geometric.typing import Adj
from torch_geometric.utils import structured_negative_sampling

from sklearn.preprocessing import OneHotEncoder


def hetergeneous_negative_sampling_naive(edge_index: Adj,
                                         node_idx_type: torch.Tensor,
                                         random_seed: int = 0) -> tuple:
    """
    alteration of torch_geometric.utils.structured_negative_sampling to accommodate for
    heterogeneous graphs
    :param edge_index: edge index of the graph/batch
    :param node_idx_type: tensor, node_id_type[i] = type of the node with id = i
    :param random_seed: reproducibility
    :return: tensor [3xn_edges] containing node (central, positive, negative) triplets
    """
    torch.manual_seed(random_seed)
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
    return np.concatenate((node_features, type_feats), axis=1)
