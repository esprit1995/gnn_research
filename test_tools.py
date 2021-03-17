import torch
import torch.nn.functional as F
from utils.losses import push_pull_metapath_instance_loss
from utils.tools import sample_metapath_instances


def test_push_pull_loss():
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
    metapath = ['0', '1', '2']
    pos_instances = [(0, 1, 2), (3, 4, 5), (7, 6, 5), (8, 6, 5)]
    neg_instances = [(0, 4, 2), (3, 1, 2), (7, 4, 2), (8, 1, 5)]

    loss_from_func = push_pull_metapath_instance_loss(pos_instances, neg_instances, toy_graph['node_features'])

    # building all the parts manually to compare the losses
    left_part_pos = torch.tensor([[1, 1, 1],
                                  [1, 1, 1],
                                  [4, 1, 1],
                                  [4, 1, 1],
                                  [8, 1, 1],
                                  [8, 1, 1],
                                  [9, 1, 1],
                                  [9, 1, 1],
                                  [2, 1, 1],
                                  [5, 1, 1],
                                  [7, 1, 1],
                                  [7, 1, 1]]).double()
    right_part_pos = torch.tensor([[2, 1, 1],
                                   [3, 1, 1],
                                   [5, 1, 1],
                                   [6, 1, 1],
                                   [7, 1, 1],
                                   [6, 1, 1],
                                   [7, 1, 1],
                                   [6, 1, 1],
                                   [3, 1, 1],
                                   [6, 1, 1],
                                   [6, 1, 1],
                                   [6, 1, 1]]).double()
    left_part_neg = torch.tensor([[1, 1, 1],
                                  [1, 1, 1],
                                  [4, 1, 1],
                                  [4, 1, 1],
                                  [8, 1, 1],
                                  [8, 1, 1],
                                  [9, 1, 1],
                                  [9, 1, 1],
                                  [5, 1, 1],
                                  [2, 1, 1],
                                  [5, 1, 1],
                                  [2, 1, 1]]).double()
    right_part_neg = torch.tensor([[5, 1, 1],
                                   [3, 1, 1],
                                   [2, 1, 1],
                                   [3, 1, 1],
                                   [5, 1, 1],
                                   [3, 1, 1],
                                   [2, 1, 1],
                                   [6, 1, 1],
                                   [3, 1, 1],
                                   [3, 1, 1],
                                   [3, 1, 1],
                                   [6, 1, 1]]).double()
    path_length = 3
    out_p = F.logsigmoid(
        torch.bmm(left_part_pos.view(int(path_length * (path_length - 1) / 2) * len(pos_instances), 1, -1),
                  right_part_pos.view(int(path_length * (path_length - 1) / 2) * len(pos_instances), -1, 1)))
    out_n = F.logsigmoid(
        -torch.bmm(left_part_neg.view(int(path_length * (path_length - 1) / 2) * len(neg_instances), 1, -1),
                   right_part_neg.view(int(path_length * (path_length - 1) / 2) * len(neg_instances), -1, 1)))
    loss_manual = - out_p.mean() - out_n.mean()
    assert((loss_manual-loss_from_func)/(loss_manual+1e-7) < 0.0001)
