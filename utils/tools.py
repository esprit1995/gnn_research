import datetime
import torch
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from torch_geometric.typing import Adj
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder

from typing import Dict, Tuple, Any
from termcolor import cprint
from functools import partial


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


def edge_index_to_adj_dict(edge_index: Dict, node_type_mask: torch.tensor, between_types: Tuple) -> Dict:
    """
    creates a adjacency dictionary between the specified node types based on the passed edge_index
    :param edge_index: a dict containing edges. edge_index[(type_1, type_2)] contains edges between
                       nodes of types type_1, type_2
    :param node_type_mask: numpy array containing node types, node_types[i] = type of node with index i.
    :param between_types: tuple (type_1, type_2); should be on of the keys of edge_index
    :return: adjacency dictionary between (type_1, type_2)
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


def edge_index_to_neg_adj_dict(edge_index: Dict, node_type_mask: torch.tensor, between_types: Tuple) -> Dict:
    """
    creates a negative adjacency dictionary between the specified node types based on the passed edge_index
    :param edge_index: a dict containing edges. edge_index[(type_1, type_2)] contains edges between
                       nodes of types type_1, type_2
    :param node_type_mask: numpy array containing node types, node_types[i] = type of node with index i.
    :param between_types: tuple (type_1, type_2); should be on of the keys of edge_index
    :return: negative adjacency dictionary between (type_1, type_2)
    """
    if between_types not in list(edge_index.keys()):
        raise ValueError("edge_index_to_neg_adj_dict(): type pair not found in the edge index")

    typed_edge_index = edge_index[between_types].numpy()
    head_type_range = np.sort(np.where(node_type_mask == int(between_types[0]))[0])
    tail_type_range = np.sort(np.where(node_type_mask == int(between_types[1]))[0])
    neg_adj_dict = dict()

    for head_type_idx in head_type_range:
        if head_type_idx in typed_edge_index[0]:
            positive_edges = typed_edge_index[1][np.where(typed_edge_index[0] == head_type_idx)].tolist()
            negative_edges = np.array(list(set(tail_type_range.tolist()).difference(set(positive_edges))))
            neg_adj_dict[str(head_type_idx)] = negative_edges
        else:
            neg_adj_dict[str(head_type_idx)] = tail_type_range
    return neg_adj_dict


def make_step(current: int, adj_dict) -> int:
    try:
        if adj_dict[str(int(current))].size > 0:
            return np.random.choice(adj_dict[str(current)])
        else:
            return -1
    except Exception as e:
        raise e


def sample_metapath_instance(metapath_: tuple,
                             adj_dicts_: dict,
                             starting_points_: np.array,
                             random_seed: int = 69,
                             stepping_method=make_step, ) -> tuple:
    """
    sample an instance for a given metapath template
    :param metapath_: metapath template. For instance, if available node types are
                      'A', 'P', 'C', it could be ('A', 'P', 'A')
    :param stepping_method: function to make random steps withing adjacency dictionaries
    :param adj_dicts_: adjacency dictionaries between different node types
    :param starting_points_: possible starting points, ndarray containing node ids.
    :param random_seed:
    :return:
    """
    metapath_instance = list()
    np.random.seed(random_seed)
    metapath_current = np.random.choice(starting_points_)
    metapath_instance.append(metapath_current)
    for nstep in range(1, len(metapath_)):
        edge_type = (metapath_[nstep - 1], metapath_[nstep])
        try:
            metapath_current = stepping_method(metapath_current, adj_dicts_[edge_type])
        except Exception as e:
            print('Sampling process failed due to exception: ' + str(e))
            raise e
        if metapath_current == -1:
            return None
        metapath_instance.append(metapath_current)
    return tuple(metapath_instance)


def sample_metapath_instances(metapath: Tuple, n: int, pyg_graph_info: Any,
                              nworkers: int = 4, parallel: bool = False,
                              negative_samples: bool = False) -> list:
    """
    sample a predefined number of metapath instances in a given graph
    :param metapath: a tuple describing a metapath. For instance, ('0', '1', '2', '1', '0')
    :param n: how many instances to randomly sample from a graph
    :param pyg_graph_info: variable containing at least 'edge_index_dict', 'node_type_mask'
    :param nworkers: parallel processing param
    :param parallel: whether to use parallel processing
    :param negative_samples: whether to sample positive or negative instances
    :return: a list of sampled metapath instances. only unique instances are kept
    """
    np.random.seed(69)
    # get all possible starting points for the given metapath template
    starting_points = np.where(pyg_graph_info['node_type_mask'].numpy() == int(metapath[0]))[0]

    # get adjacency dictionaries needed for the given metapath template
    adj_dicts = dict()
    for nedge in range(1, len(metapath)):
        if not negative_samples:
            adj_dicts[(metapath[nedge - 1], metapath[nedge])] = edge_index_to_adj_dict(
                pyg_graph_info['edge_index_dict'],
                pyg_graph_info['node_type_mask'],
                (metapath[nedge - 1],
                 metapath[nedge]))
        else:
            adj_dicts[(metapath[nedge - 1], metapath[nedge])] = edge_index_to_neg_adj_dict(
                pyg_graph_info['edge_index_dict'],
                pyg_graph_info['node_type_mask'],
                (metapath[nedge - 1],
                 metapath[nedge]))

    # instance sampling
    if parallel:
        pool = mp.Pool(processes=nworkers)
        results = [pool.apply_async(sample_metapath_instance, args=(metapath, adj_dicts, starting_points, i)) for i in
                   range(n)]
        mp_instances = [p.get() for p in results]
        return list(set(mp_instances))
    else:
        results = list()
        for i in range(n):
            instance = sample_metapath_instance(metapath, adj_dicts, starting_points, i)
            if instance is not None:
                results.append(sample_metapath_instance(metapath, adj_dicts, starting_points, i))
        return list(set(results))


def corrupt_positive_metapath_instance(mpinstance: tuple,
                                       mptemplate: tuple,
                                       positions: tuple,
                                       adj_dicts: dict,
                                       neg_adj_dicts: dict,
                                       node_type_mask: torch.tensor,
                                       method: str = 'random'):
    """
    corrupt positive metapath instance by replacing the nodes indicated by __positions__ argument
    in one of the 2 ways: replace with random negatives, replace with part of another metapath instance
    :param mpinstance: a metapath instance, a tuple containing node ids
    :param mptemplate: a metapath template, a tuple like ('1', '2', '1')
    :param positions: positions where to corrupt the instance, tuple (min_index, max_index).
                      a) if min_index == max_index, corrupt in just one spot
                      b) else, corrupt between min_index and max_index inclusively
    :param adj_dicts: precomputed adjacency dictionaries
    :param neg_adj_dicts: precomputed negative adjacency dictionaries
    :param node_type_mask: tensor encoding node types
    :param method: 'random' or 'crossover'
    :return: tuple - corrupted mptemplate instance or None in case of problems
    """
    assert positions[1] < len(mptemplate) or positions[0] > 0, \
        'corrupt_positive_metapath_instance(): invalid positions argument, out of range'
    corrupted_instance = list(mpinstance)

    # corrupt with just some random nodes
    if method == 'random':
        for indx in range(positions[0], positions[1] + 1):
            if indx == 0:
                candidates = np.where(node_type_mask.numpy() == int(mptemplate[indx]))[0]
                corrupted_instance[indx] = np.random.choice(
                    np.delete(np.argwhere(candidates == mpinstance[0]),
                              candidates,
                              0))
            else:
                transition_type = (mptemplate[indx - 1], mptemplate[indx])
                corrupted_instance[indx] = make_step(mpinstance[indx - 1],
                                                     neg_adj_dicts[transition_type])

    # corrupt by doing a crossover with another positive instance
    elif method == 'crossover':
        for indx in range(positions[0], positions[1] + 1):
            if indx == 0:
                candidates = np.where(node_type_mask.numpy() == int(mptemplate[indx]))[0]
                corrupted_instance[indx] = np.random.choice(
                    np.delete(np.argwhere(candidates == mpinstance[0]),
                              candidates,
                              0))
            elif indx == positions[0]:
                transition_type = (mptemplate[indx - 1], mptemplate[indx])
                corrupted_instance[indx] = make_step(mpinstance[indx - 1],
                                                     neg_adj_dicts[transition_type])
            else:
                transition_type = (mptemplate[indx - 1], mptemplate[indx])
                corrupted_instance[indx] = make_step(corrupted_instance[indx - 1],
                                                     adj_dicts[transition_type])
    else:
        raise NotImplementedError('corrupt_positive_metapath_instance(): requested corruption method unimplemented')
    return tuple(corrupted_instance)


def IMDB_DBLP_ACM_metapath_instance_sampler(dataset, metapath: Tuple, n: int,
                                            corruption_method: str = 'random',
                                            corruption_position: tuple = (0, 0)) -> tuple:
    """
    sampler wrapper for IMDB_DBLP_ACM dataset
    :param dataset: dataset from which to sample the instances
    :param metapath: tuple containing metapath template
    :param n: how many instances to sample
    :param corruption_method: 'random' or 'crossover'
    :param corruption_position: tuple (idx_min, idx_max)
    :return: (positive_instances, corrupted instances))
    """
    ds = dataset
    adj_dicts = dict()
    neg_adj_dicts = dict()
    for key in list(ds['edge_index_dict'].keys()):
        adj_dicts[key] = edge_index_to_adj_dict(ds['edge_index_dict'], ds['node_type_mask'], key)
        neg_adj_dicts[key] = edge_index_to_neg_adj_dict(ds['edge_index_dict'], ds['node_type_mask'], key)

    positive_instances = sample_metapath_instances(metapath, n, ds, negative_samples=False)
    negative_instances = list()
    for i in range(len(positive_instances)):
        negative_instances.append(corrupt_positive_metapath_instance(mpinstance=positive_instances[i],
                                                                     mptemplate=metapath,
                                                                     positions=corruption_position,
                                                                     adj_dicts=adj_dicts,
                                                                     neg_adj_dicts=neg_adj_dicts,
                                                                     node_type_mask=ds['node_type_mask'],
                                                                     method=corruption_method))
    return positive_instances, negative_instances


# ############################################################
# ############################################################
# ############################################################

# ############################################################
# #####  Graphlet-sampling: general graphlets          #######
# ############################################################

# # example of a graphlet definition: for DBLP graphlet
#                     A
#                     |
#               A  -  P  -  A
#                     |
#                     C
# # we define its template like so:
# new_graphlet = {'main': ['A', 'P', 'A'], ---> __longest__ linear component is the main lane
#                 'sub_paths':  {
#                            1: [['P', 'A'], ['P', 'C']]
#                            }}
#                            |
#                at mane lane index 1, i.e. at node 'P', 2 sub-lanes start: 'PA' and 'PC'
# sampled graphlets will be keep ids, as, in our example:
# new_graphlet_sample = {'main': [100, 101, 4],
#                        'sub_paths': {
#                               1: [[101, 200], [101, 10]]
#                               },
#                        'node_ids_present': [100, 101, 4, 200, 10] ---> indicates which nodes are used in this instance}

def make_step_restricted(current: int, adj_dict__: dict, to_avoid: list):
    """
    makes metapath-sampling step while avoiding node ids listed in to_avoid
    :param current: id of the node metapath is currently at
    :param adj_dict__: adjacency dictionary
    :param to_avoid: list of node ids to avoid
    :return:
    """
    try:
        if adj_dict__[str(int(current))].size > 0:
            chosen_node = np.random.choice(adj_dict__[str(current)])
            attempts = 0
            while chosen_node in to_avoid:
                chosen_node = np.random.choice(adj_dict__[str(current)])
                attempts += 1
                if attempts > 20:
                    return -1
            return chosen_node
        else:
            return -1
    except Exception as e:
        raise e


def sample_graphlet_instance(graphlet_template: dict,
                             adj_dicts,
                             starting_points,
                             random_seed: int = 69):
    """
    sample a graphlet instance for a given template
    :param graphlet_template: graphlet template as described above
    :param adj_dicts: adjacency dictionaries between different node types
    :param starting_points: where to start the main lane of the graphlet
    :param random_seed: reproducibility
    :return:
    """
    # adj_dicts_copy = deepcopy(adj_dicts)
    graphlet_instance = dict()
    # sample the main lane using metapath sampling function
    graphlet_instance['main'] = list(sample_metapath_instance(metapath_=tuple(graphlet_template['main']),
                                                              adj_dicts_=adj_dicts,
                                                              starting_points_=starting_points,
                                                              random_seed=random_seed,
                                                              stepping_method=make_step))
    graphlet_instance['node_ids_present'] = [] + graphlet_instance['main']
    # sample sub paths
    graphlet_instance['sub_paths'] = dict()
    for mainlane_start_indx in list(graphlet_template['sub_paths'].keys()):
        subpath_instances = []
        for metapath_template in graphlet_template['sub_paths'][mainlane_start_indx]:
            sub_path = sample_metapath_instance(metapath_=tuple(metapath_template),
                                                adj_dicts_=adj_dicts,
                                                starting_points_=np.array(
                                                    [graphlet_instance['main'][int(mainlane_start_indx)]]),
                                                random_seed=random_seed,
                                                stepping_method=partial(make_step_restricted,
                                                                        to_avoid=graphlet_instance[
                                                                            'node_ids_present']))
            if sub_path is None:
                return None
            subpath_instances.append(list(sub_path))
            graphlet_instance['node_ids_present'] = list(graphlet_instance['node_ids_present'] + subpath_instances[-1])
        graphlet_instance['sub_paths'][mainlane_start_indx] = subpath_instances
    return graphlet_instance


def remove_duplicate_graphlets(graphlet_list):
    node_lists = [graphlet['node_ids_present'] for graphlet in graphlet_list]
    hash_list = [''.join([str(nodelist[i]) for i in range(len(nodelist))]) for nodelist in node_lists]

    # find out which indices to keep
    encountered = []
    tokeep_index = []
    for index, elem in enumerate(hash_list):
        if elem in encountered:
            continue
        else:
            encountered.append(elem)
            tokeep_index.append(index)
    res = [graphlet_list[i] for i in range(len(graphlet_list)) if i in tokeep_index]
    return res


def sample_n_graphlet_instances(graphlet_template: dict, graph_info: Any, n_samples: int = 1):
    """
    sample n instances for a given graphlet template
    :param graphlet_template: template for a graphlet as it is described above
    :param graph_info: object containing graph information. At least fields 'edge_index_dict' and
                        'node_type_mask' are required
    :param n_samples: how many samples to return
    :return:
    """
    np.random.seed(69)
    # get all possible starting points for the given metapath template
    starting_node_type = graphlet_template['main'][0]
    starting_points = np.where(graph_info['node_type_mask'].numpy() == int(starting_node_type))[0]

    # get adjacency dictionaries
    adj_dicts = dict()
    possible_edge_types = list(graph_info['edge_index_dict'].keys())
    for edge_type in possible_edge_types:
        adj_dicts[edge_type] = edge_index_to_adj_dict(
            graph_info['edge_index_dict'],
            graph_info['node_type_mask'],
            edge_type)
    results = list()
    for i in range(n_samples):
        graphlet_instance = sample_graphlet_instance(graphlet_template, adj_dicts, starting_points, i)
        if graphlet_instance is not None:
            results.append(graphlet_instance)
    return remove_duplicate_graphlets(list(results))


# ############################################################
# ############        Miscellaneous           ################
# ############################################################

def label_dict_to_metadata(label_dict: dict):
    """
    dictionary is expected to contain node ids and their labels
    :param label_dict: {'key': [[node ids], [labels]]
    :return:
    """
    all_keys = [elem for elem in list(label_dict.keys())]
    ids_labels = label_dict[all_keys[0]]
    for idx in range(1, len(all_keys)):
        ids_labels = np.vstack([ids_labels, label_dict[all_keys[idx]]])
    return ids_labels[:, 0], ids_labels[:, 1]


# ----------------- helper funcs for NSHE
def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if len(sparse_mx.nonzero()[0]) == 0:
        # 空矩阵
        r, c = sparse_mx.shape
        return torch.sparse.FloatTensor(r, c)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
