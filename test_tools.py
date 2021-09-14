import torch
import numpy as np

import torch.nn.functional as F
from datasets import DBLP_ACM_IMDB_from_NSHE
from utils.losses import push_pull_metapath_instance_loss, push_pull_metapath_instance_loss_tf, push_pull_graphlet_instance_loss
from utils.tools import sample_metapath_instances, corrupt_positive_metapath_instance
from utils.tools import edge_index_to_adj_dict, edge_index_to_neg_adj_dict
from utils.tools import sample_n_graphlet_instances, corrupt_positive_graphlet_instance

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf


def test_instance_corruption():
    toy_graph = dict()
    toy_graph['node_type_mask'] = torch.tensor([0, 1, 2, 0, 1, 2, 1, 0, 0, 1, 0])
    toy_graph['edge_index_dict'] = {('0', '1'): torch.tensor([[0, 3, 7, 8, 10], [1, 4, 6, 6, 9]]),
                                    ('1', '2'): torch.tensor([[1, 4, 6, 9], [2, 5, 5, 5]]),
                                    ('1', '0'): torch.tensor([[1, 4, 6, 6, 9], [0, 3, 7, 8, 10]]),
                                    ('2', '1'): torch.tensor([[2, 5, 5, 5], [1, 4, 6, 9]])}

    metapath = ['0', '1', '2', '1', '0']
    adj_dicts = dict()
    neg_adj_dict = dict()
    for adj_type in [('0', '1'), ('1', '2'), ('2', '1'), ('1', '0')]:
        adj_dicts[adj_type] = edge_index_to_adj_dict(toy_graph['edge_index_dict'],
                                                     toy_graph['node_type_mask'],
                                                     adj_type)
        neg_adj_dict[adj_type] = edge_index_to_neg_adj_dict(toy_graph['edge_index_dict'],
                                                            toy_graph['node_type_mask'],
                                                            adj_type)
    mpinstance1 = (3, 4, 5, 9, 10)
    positions1 = (len(mpinstance1) - 1, len(mpinstance1) - 1)
    corrupted_instance1 = corrupt_positive_metapath_instance(mpinstance1,
                                                             tuple(metapath),
                                                             positions1,
                                                             adj_dicts,
                                                             neg_adj_dict,
                                                             toy_graph['node_type_mask'],
                                                             method='random')

    mpinstance2 = (3, 4, 5, 9, 10)
    positions2 = (len(mpinstance2) - 2, len(mpinstance2) - 1)
    corrupted_instance2 = corrupt_positive_metapath_instance(mpinstance2,
                                                             tuple(metapath),
                                                             positions2,
                                                             adj_dicts,
                                                             neg_adj_dict,
                                                             toy_graph['node_type_mask'],
                                                             method='crossover')
    cond1 = corrupted_instance1[4] in [0, 3, 7, 8]
    cond2 = corrupted_instance2[3] == 6 and corrupted_instance2[4] == 7 \
            or corrupted_instance2[3] == 7 and corrupted_instance2[4] == 8 \
            or corrupted_instance2[3] == 1 and corrupted_instance2[4] == 0
    assert cond1 and cond2


def test_push_pull_metapath_loss():
    toy_graph = dict()
    toy_graph['node_type_mask'] = torch.tensor([0, 1, 2, 0, 1, 2, 1, 0, 0])
    toy_graph['edge_index_dict'] = {('0', '1'): torch.tensor([[0, 3, 7, 8], [1, 4, 6, 6]]).double(),
                                    ('1', '2'): torch.tensor([[1, 4, 6], [2, 5, 5]]).double(),
                                    ('1', '0'): torch.tensor([[1, 4, 6, 6], [0, 3, 7, 8]]).double(),
                                    ('2', '1'): torch.tensor([[2, 5, 5], [1, 4, 6]]).double()}
    toy_graph['node_features'] = torch.tensor([[1, 1, 1],
                                               [2, 1, 1],
                                               [3, 1, 1],
                                               [4, 1, 1],
                                               [5, 1, 1],
                                               [6, 1, 1],
                                               [7, 1, 1],
                                               [8, 1, 1],
                                               [9, 1, 1]]).double()
    metapath = ['0', '1', '2', '1']
    pos_instances = [(3, 4, 5, 6), (7, 6, 5, 4)]
    neg_instances = [(3, 1, 2, 6), (7, 1, 2, 4)]
    corrupted_positions = (1, 2)
    loss_from_func = push_pull_metapath_instance_loss(pos_instances,
                                                      neg_instances, corrupted_positions,
                                                      toy_graph['node_features'])

    # building all the parts manually to compare the losses
    left_part_pos = torch.tensor([[4, 1, 1],
                                  [4, 1, 1],
                                  [4, 1, 1],
                                  [5, 1, 1],
                                  [5, 1, 1],
                                  [6, 1, 1],
                                  [8, 1, 1],
                                  [8, 1, 1],
                                  [8, 1, 1],
                                  [7, 1, 1],
                                  [7, 1, 1],
                                  [6, 1, 1]]).double()
    right_part_pos = torch.tensor([[5, 1, 1],
                                   [6, 1, 1],
                                   [7, 1, 1],
                                   [6, 1, 1],
                                   [7, 1, 1],
                                   [7, 1, 1],
                                   [7, 1, 1],
                                   [6, 1, 1],
                                   [5, 1, 1],
                                   [6, 1, 1],
                                   [5, 1, 1],
                                   [5, 1, 1]]).double()
    left_part_neg = torch.tensor([[2, 1, 1],
                                  [2, 1, 1],
                                  [3, 1, 1],
                                  [3, 1, 1],
                                  [2, 1, 1],
                                  [2, 1, 1],
                                  [3, 1, 1],
                                  [3, 1, 1]]).double()
    right_part_neg = torch.tensor([[4, 1, 1],
                                   [7, 1, 1],
                                   [4, 1, 1],
                                   [7, 1, 1],
                                   [8, 1, 1],
                                   [5, 1, 1],
                                   [8, 1, 1],
                                   [5, 1, 1]]).double()
    path_length = 4
    out_p = F.logsigmoid(
        torch.bmm(left_part_pos.view(int(path_length * (path_length - 1) / 2) * len(pos_instances), 1, -1),
                  right_part_pos.view(int(path_length * (path_length - 1) / 2) * len(pos_instances), -1, 1)))
    out_n = F.logsigmoid(
        -torch.bmm(left_part_neg.view(8, 1, -1),
                   right_part_neg.view(8, -1, 1)))
    loss_manual = - out_p.mean() - out_n.mean()
    assert ((loss_manual - loss_from_func) / (loss_manual + 1e-7) < 0.0001)


def test_push_pull_graphlet_loss():
    toy_graph = dict()
    toy_graph['node_type_mask'] = torch.tensor([0, 1, 2, 0, 1, 2, 1, 0, 0])
    toy_graph['edge_index_dict'] = {('0', '1'): torch.tensor([[0, 4, 5, 6, 9, 10, 4], [1, 3, 1, 3, 8, 8, 11]]).double(),
                                    ('1', '0'): torch.tensor([[1, 3, 1, 3, 8, 8, 11], [0, 4, 5, 6, 9, 10, 4]]).double(),
                                    ('1', '2'): torch.tensor([[1, 3, 8], [2, 2, 7]]).double(),
                                    ('2', '1'): torch.tensor([[2, 2, 7], [1, 3, 8]]).double()}
    toy_graph['node_features'] = torch.tensor([[1, 1, 1],
                                               [2, 1, 1],
                                               [3, 1, 1],
                                               [4, 1, 1],
                                               [5, 1, 1],
                                               [6, 1, 1],
                                               [7, 1, 1],
                                               [8, 1, 1],
                                               [9, 1, 1],
                                               [10, 1, 1],
                                               [11, 1, 1],
                                               [12, 1, 1]]).double()
    pos_instances = [{'main': [0, 1, 2, 3, 4],
                      'sub_paths': {1: [[1, 5]],
                                    3: [[3, 6]]},
                      'node_ids_present': [0, 1, 2, 3, 4, 5, 6]},
                     {'main': [6, 3, 7, 8, 10],
                      'sub_paths': {1: [[3, 4]],
                                    3: [[8, 9]]},
                      'node_ids_present': [6, 3, 7, 8, 10, 4, 9]}]
    cor_instances = [{'main': [0, 1, 7, 11, 4],
                      'sub_paths': {1: [[1 ,5]],
                                    3: [[11, 10]]},
                      'original_node_ids_present': [0, 1, 2, 3, 4, 5, 6]},
                     {'main': [6, 3, 2, 11, 10],
                      'sub_paths': {1: [[3, 4]],
                                    3: [[11, 0]]},
                      'original_node_ids_present': [6, 3, 7, 8, 10, 4, 9]}]
    loss_from_func = push_pull_graphlet_instance_loss(pos_instances, cor_instances, node_embeddings=toy_graph['node_features'])

    # manually calculating the loss to compare the results
    left_part_pos = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
                                  [2, 1, 1], [2, 1, 1], [2, 1, 1], [2, 1, 1], [2, 1, 1],
                                  [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1],
                                  [4, 1, 1], [4, 1, 1], [4, 1, 1],
                                  [5, 1, 1], [5, 1, 1],
                                  [6, 1, 1],
                                  [4, 1, 1], [4, 1, 1], [4, 1, 1], [4, 1, 1], [4, 1, 1], [4, 1, 1],
                                  [5, 1, 1], [5, 1, 1], [5, 1, 1], [5, 1, 1], [5, 1, 1],
                                  [7, 1, 1], [7, 1, 1], [7, 1, 1], [7, 1, 1],
                                  [8, 1, 1], [8, 1, 1], [8, 1, 1],
                                  [9, 1, 1], [9, 1, 1],
                                  [10, 1, 1]]).double()

    right_part_pos = torch.tensor([[2, 1, 1], [3, 1, 1], [4, 1, 1], [5, 1, 1], [6, 1, 1], [7, 1, 1],
                                   [3, 1, 1], [4, 1, 1], [5, 1, 1], [6, 1, 1], [7, 1, 1],
                                   [4, 1, 1], [5, 1, 1], [6, 1, 1], [7, 1, 1],
                                   [5, 1, 1], [6, 1, 1], [7, 1, 1],
                                   [6, 1, 1], [7, 1, 1],
                                   [7, 1, 1],
                                   [5, 1, 1], [7, 1, 1], [8, 1, 1], [9, 1, 1], [10, 1, 1], [11, 1, 1],
                                   [7, 1, 1], [8, 1, 1], [9, 1, 1], [10, 1, 1], [11, 1, 1],
                                   [8, 1, 1], [9, 1, 1], [10, 1, 1], [11, 1, 1],
                                   [9, 1, 1], [10, 1, 1], [11, 1, 1],
                                   [10, 1, 1], [11, 1, 1],
                                   [11, 1, 1]]).double()
    left_part_neg = torch.tensor([[1, 1, 1], [2, 1, 1], [5, 1, 1], [6, 1, 1],
                                  [1, 1, 1], [2, 1, 1], [5, 1, 1], [6, 1, 1],
                                  [1, 1, 1], [2, 1, 1], [5, 1, 1], [6, 1, 1],
                                  [4, 1, 1], [5, 1, 1], [7, 1, 1], [11, 1, 1],
                                  [4, 1, 1], [5, 1, 1], [7, 1, 1], [11, 1, 1],
                                  [4, 1, 1], [5, 1, 1], [7, 1, 1], [11, 1, 1]]).double()
    right_part_neg = torch.tensor([[8, 1, 1], [8, 1, 1], [8, 1, 1], [8, 1, 1],
                                  [12, 1, 1], [12, 1, 1], [12, 1, 1], [12, 1, 1],
                                  [11, 1, 1], [11, 1, 1], [11, 1, 1], [11, 1, 1],
                                  [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1],
                                  [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
                                  [12, 1, 1], [12, 1, 1], [12, 1, 1], [12, 1, 1]]).double()

    path_length = 7
    out_p = F.logsigmoid(
        torch.bmm(left_part_pos.view(int(path_length * (path_length - 1) / 2) * len(pos_instances), 1, -1),
                  right_part_pos.view(int(path_length * (path_length - 1) / 2) * len(pos_instances), -1, 1)))
    out_n = F.logsigmoid(
        -torch.bmm(left_part_neg.view(24, 1, -1),
                   right_part_neg.view(24, -1, 1)))
    loss_manual = - out_p.mean() - out_n.mean()
    assert ((loss_manual - loss_from_func) / (loss_manual + 1e-7) < 0.0001)

def test_tf_pt_loss_correspondence():
    toy_graph = dict()
    toy_graph['node_type_mask'] = torch.tensor([0, 1, 2, 0, 1, 2, 1, 0, 0])
    toy_graph['edge_index_dict'] = {('0', '1'): torch.tensor([[0, 3, 7, 8], [1, 4, 6, 6]]).double(),
                                    ('1', '2'): torch.tensor([[1, 4, 6], [2, 5, 5]]).double(),
                                    ('1', '0'): torch.tensor([[1, 4, 6, 6], [0, 3, 7, 8]]).double(),
                                    ('2', '1'): torch.tensor([[2, 5, 5], [1, 4, 6]]).double()}
    toy_graph['node_features'] = torch.tensor([[1, 1, 1],
                                               [2, 1, 1],
                                               [3, 1, 1],
                                               [4, 1, 1],
                                               [5, 1, 1],
                                               [6, 1, 1],
                                               [7, 1, 1],
                                               [8, 1, 1],
                                               [9, 1, 1]]).double()
    metapath = ['0', '1', '2', '1']
    pos_instances = [(3, 4, 5, 6), (7, 6, 5, 4)]
    neg_instances = [(3, 1, 2, 6), (7, 1, 2, 4)]
    corrupted_positions = (1, 2)

    loss_from_torch = push_pull_metapath_instance_loss(pos_instances,
                                                       neg_instances, corrupted_positions,
                                                       toy_graph['node_features'])
    loss_from_tf = push_pull_metapath_instance_loss_tf(pos_instances,
                                                       neg_instances, corrupted_positions,
                                                       tf.convert_to_tensor(toy_graph['node_features'].numpy()))

    difference = loss_from_tf.eval(session=tf.compat.v1.Session()).reshape(-1)[0] - loss_from_torch.item()
    assert (abs(difference / loss_from_torch.item()) +
            abs(difference / loss_from_tf.eval(session=tf.compat.v1.Session()).reshape(-1)[0]) < 0.00001)
    return True


def test_sample_graphlet_instances():
    template = {'main': ['0', '1', '2'],
                'sub_paths': {
                    1: [['1', '0']]
                }}
    graph_info = dict()
    graph_info['node_type_mask'] = torch.tensor([0, 1, 2, 1, 0, 0, 0, 0])
    graph_info['edge_index_dict'] = {('0', '1'): torch.tensor([[0, 5, 6, 4, 7], [1, 1, 3, 3, 3]]),
                                     ('1', '0'): torch.tensor([[1, 1, 3, 3, 3], [0, 5, 6, 4, 7]]),
                                     ('1', '2'): torch.tensor([[1, 3], [2, 2]]),
                                     ('2', '1'): torch.tensor([[2, 2], [1, 3]])}
    sampled_graphlets = sample_n_graphlet_instances(template, graph_info, n_samples=20)
    return sampled_graphlets


def test_graphlet_instance_corruption():
    graph_info = dict()
    graph_info['node_type_mask'] = torch.tensor([0, 1, 2, 1, 0, 0, 0, 0])
    graph_info['edge_index_dict'] = {('0', '1'): torch.tensor([[0, 5, 6, 4, 7], [1, 1, 3, 3, 3]]),
                                     ('1', '0'): torch.tensor([[1, 1, 3, 3, 3], [0, 5, 6, 4, 7]]),
                                     ('1', '2'): torch.tensor([[1, 3], [2, 2]]),
                                     ('2', '1'): torch.tensor([[2, 2], [1, 3]])}
    template = {'main': ['0', '1', '2'],
                'sub_paths': {
                    1: [['1', '0']]
                }}
    instance = {'main': [0, 1, 2],
                'sub_paths': {
                    1: [[1, 5]]
                },
                'node_ids_present': [0, 1, 2, 5]}
    corr_positions = {'main': (0, 1),
                      'sub_paths': {
                          1: [(0, 0)]
                      }}
    node_type_mask = torch.tensor([0, 1, 2, 1, 0, 0, 0, 0])
    neg_adj_dict = dict()
    for adj_type in [('0', '1'), ('1', '2'), ('2', '1'), ('1', '0')]:

        neg_adj_dict[adj_type] = edge_index_to_neg_adj_dict(graph_info['edge_index_dict'],
                                                            graph_info['node_type_mask'],
                                                            adj_type)

    result = corrupt_positive_graphlet_instance(instance, template, corr_positions,
                                                neg_adj_dict, node_type_mask)
    return result


if __name__ == "__main__":
    # ds = DBLP_ACM_IMDB_from_NSHE(name='dblp', root='/home/ubuntu/msandal_code/PyG_playground/data/NSHE')[0]
    # graphlet_template = {'main': ['0', '1', '2', '1'], 'sub_paths': {0: [['0', '1', '2']]}}
    # graphlets = sample_n_graphlet_instances(graphlet_template, ds, n_samples=10000)
    test_push_pull_graphlet_loss()
