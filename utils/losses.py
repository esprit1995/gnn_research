import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf


# -----------------------------------------------------------
#                   LOSS FUNCTIONS
# -----------------------------------------------------------

def triplet_loss_pure(id_triplets: Tuple[torch.tensor, torch.tensor, torch.tensor],
                      node_embeddings: torch.tensor):
    """
    unsupervised triplet loss
    :param id_triplets: tuple([central], [positive], [negative]). All elements are torch.tensors
    :param node_embeddings: tensor containing current node embeddings
    :return: triplet loss
    """
    assert id_triplets[0].shape[0] == id_triplets[1].shape[0] and id_triplets[1].shape[0] == id_triplets[2].shape[0], \
        "triplet_loss_pure(): triplet lists of unequal length!"

    batch_size = id_triplets[0].shape[0]
    emb_dim = node_embeddings.shape[1]
    central = node_embeddings[id_triplets[0]].view(batch_size, 1, emb_dim)
    positive = node_embeddings[id_triplets[1]].view(batch_size, emb_dim, 1)
    negative = node_embeddings[id_triplets[2]].view(batch_size, emb_dim, 1)

    out_p = torch.bmm(central, positive)
    out_n = - torch.bmm(central, negative)

    return (-(F.logsigmoid(out_p) + F.logsigmoid(out_n))).mean()


def triplet_loss_type_aware(id_triplets: Tuple[torch.tensor, torch.tensor, torch.tensor],
                            node_embeddings: torch.tensor,
                            id_type_mask: torch.tensor,
                            lmbd: float):
    """
    unsupervised type-aware triplet loss
    !! expects that triplets have positive and negative samples OF THE SAME TYPE! !!
    :param id_triplets: tuple([central], [positive], [negative]). All elements are torch.tensors
    :param node_embeddings: tensor containing current node embeddings
    :param id_type_mask: tensor containing type code of node idx i at position i
    :param lmbd: how strong the type loss is supposed to be
    :return: triplet loss
    """
    assert id_triplets[0].shape[0] == id_triplets[1].shape[0] and id_triplets[1].shape[0] == id_triplets[2].shape[0], \
        "triplet_loss_type_aware(): triplet lists of unequal length!"

    # computing regular loss
    batch_size = id_triplets[0].shape[0]
    emb_dim = node_embeddings.shape[1]
    central = node_embeddings[id_triplets[0]].view(batch_size, 1, emb_dim)
    positive = node_embeddings[id_triplets[1]].view(batch_size, emb_dim, 1)
    negative = node_embeddings[id_triplets[2]].view(batch_size, emb_dim, 1)

    out_p = torch.bmm(central, positive)
    out_n = - torch.bmm(central, negative)

    normal_loss = (-(F.logsigmoid(out_p) + F.logsigmoid(out_n))).mean()

    # computing type loss
    type_comparison_mask = np.array(id_type_mask[id_triplets[0]] == id_type_mask[id_triplets[1]])
    central_positive = node_embeddings[id_triplets[0][type_comparison_mask]].view(-1, 1, emb_dim)
    central_negative = node_embeddings[id_triplets[0][np.logical_not(type_comparison_mask)]].view(-1, 1, emb_dim)
    type_positive = torch.cat((node_embeddings[id_triplets[1][type_comparison_mask]].view(-1, emb_dim, 1),
                               node_embeddings[id_triplets[2][type_comparison_mask]].view(-1, emb_dim, 1)), 0)
    type_negative = torch.cat(
        (node_embeddings[id_triplets[1][np.logical_not(type_comparison_mask)]].view(-1, emb_dim, 1),
         node_embeddings[id_triplets[2][np.logical_not(type_comparison_mask)]].view(-1, emb_dim, 1)), 0)
    out_p = F.logsigmoid(torch.bmm(torch.cat(2 * [central_positive], 0), type_positive))
    out_n = F.logsigmoid(- torch.bmm(torch.cat(2 * [central_negative], 0), type_negative))
    type_loss = - (out_p.mean() if out_p.numel() != 0 else 0) - (out_n.mean() if out_n.numel() != 0 else 0)
    return normal_loss + lmbd * type_loss


def push_pull_metapath_instance_loss(pos_instances, corrupted_instances,
                                     corrupted_positions, node_embeddings: torch.tensor):
    """
    compute the push-pull loss (logsigmoid loss) over the given metapath instances.
    assumes that positive and negative metapath instances have the same template
    :param pos_instances: list of tuples containing positive metapath instances
    :param corrupted_instances: list of tuples containing corrupted metapath instances
    :param corrupted_positions: tuple (min_idx, max_idx) indicating indices at which the nodes had been corrupted
                                !!!MUST BE THE SAME FOR ALL CORRUPTED INSTANCES!!! Corrupting positive instances in
                                different positions calls for the computation of a separate push-pull loss!
    :param node_embeddings: current embeddings of the nodes
    :return: push-pull loss
    """
    path_length = len(pos_instances[0])
    embed_dim = node_embeddings.shape[1]

    n_corrupted = corrupted_positions[1] - corrupted_positions[0] + 1
    n_normal = path_length - n_corrupted

    # ----- computing loss contribution from the corrupted instances
    neg_tensor = torch.tensor(corrupted_instances)
    corrupted_nodes = neg_tensor[:, corrupted_positions[0]:(corrupted_positions[1] + 1)]
    normal_nodes = neg_tensor[:, [elem for elem in range(neg_tensor.shape[1]) \
                                  if elem < corrupted_positions[0] or elem > corrupted_positions[1]]]
    # prepare tensors for computing dot product: every normal node vs every corrupted node
    corrupted_nodes = torch.repeat_interleave(corrupted_nodes, n_normal, 1).reshape(-1)
    normal_nodes = torch.repeat_interleave(normal_nodes, n_corrupted, 0).reshape(-1)
    left_part_neg = node_embeddings[corrupted_nodes]
    right_part_neg = node_embeddings[normal_nodes]
    out_n = F.logsigmoid(
        -torch.bmm(left_part_neg.view(n_corrupted * n_normal * len(pos_instances), 1, -1),
                   right_part_neg.view(n_corrupted * n_normal * len(pos_instances), -1, 1)))

    # ----- computing loss contribution from the positive instances
    pos_tensor = torch.tensor(pos_instances)
    # computing individual dot products
    left_part_pos = torch.repeat_interleave(node_embeddings[pos_tensor[:, 0]], path_length - 1, dim=0).reshape(-1,
                                                                                                               embed_dim)
    for i in range(1, path_length):
        left_part_pos = torch.vstack([left_part_pos,
                                      torch.repeat_interleave(node_embeddings[pos_tensor[:, i]],
                                                              path_length - 1 - i, dim=0).reshape(-1, embed_dim)])

    right_part_pos = torch.vstack([node_embeddings[pos_tensor[:, 1:].reshape(-1)]])
    for i in range(1, path_length - 1):
        right_part_pos = torch.vstack([right_part_pos] + [node_embeddings[pos_tensor[:, 1 + i:].reshape(-1)]])

    out_p = F.logsigmoid(
        torch.bmm(left_part_pos.view(int(path_length * (path_length - 1) / 2) * len(pos_instances), 1, -1),
                  right_part_pos.view(int(path_length * (path_length - 1) / 2) * len(pos_instances), -1, 1)))

    # ----- putting the contributions together and returning
    # pytorch-tensorflow comparison prints
    # print(out_p.reshape(-1))
    # print(out_n.reshape(-1))
    return -out_p.mean() - out_n.mean()


def push_pull_metapath_instance_loss_tf(pos_instances: list, corrupted_instances: list,
                                        corrupted_positions: tuple, node_embeddings):
    """
    compute the push-pull loss (logsigmoid loss) over the given metapath instances.
    assumes that positive and negative metapath instances have the same template
    :param pos_instances: list of tuples containing positive metapath instances
    :param corrupted_instances: list of tuples containing corrupted metapath instances
    :param corrupted_positions: tuple (min_idx, max_idx) indicating indices at which the nodes had been corrupted
                                !!!MUST BE THE SAME FOR ALL CORRUPTED INSTANCES!!! Corrupting positive instances in
                                different positions calls for the computation of a separate push-pull loss!
    :param node_embeddings: current embeddings of the nodes
    :return: push-pull loss
    """
    path_length = len(pos_instances[0])
    embed_dim = node_embeddings.shape[1].value

    n_corrupted = corrupted_positions[1] - corrupted_positions[0] + 1
    n_normal = path_length - n_corrupted

    # ----- computing loss contribution from the corrupted instances
    neg_tensor = tf.convert_to_tensor(corrupted_instances)
    corrupted_nodes = neg_tensor[:, corrupted_positions[0]:(corrupted_positions[1] + 1)]
    normal_indices = [elem for elem in range(neg_tensor.shape[1].value) \
                      if
                      elem < corrupted_positions[0] or elem > corrupted_positions[1]]
    normal_nodes = tf.gather(neg_tensor, normal_indices, axis=1)
    # prepare tensors for computing dot product: every normal node vs every corrupted node
    corrupted_nodes = tf.reshape(tf.repeat(corrupted_nodes, n_normal, 1), (-1,))
    normal_nodes = tf.reshape(tf.repeat(normal_nodes, n_corrupted, 0), (-1,))
    left_part_neg = tf.gather(node_embeddings, corrupted_nodes.eval(session=tf.compat.v1.Session()), axis=0)
    right_part_neg = tf.gather(node_embeddings, normal_nodes.eval(session=tf.compat.v1.Session()), axis=0)
    # out_n = F.logsigmoid(
    #     -torch.bmm(left_part_neg.view(n_corrupted*n_normal*len(pos_instances), 1, -1),
    #                right_part_neg.view(n_corrupted*n_normal*len(pos_instances), -1, 1)))
    out_n = tf.math.log_sigmoid(
        -tf.matmul(tf.reshape(left_part_neg, (n_corrupted * n_normal * len(pos_instances), 1, -1)),
                   tf.reshape(right_part_neg, (n_corrupted * n_normal * len(pos_instances), -1, 1))))
    # ----- computing loss contribution from the positive instances
    pos_tensor = tf.convert_to_tensor(pos_instances)
    # computing individual dot products
    left_part_pos = tf.reshape(tf.repeat(tf.gather(node_embeddings,
                                                   pos_tensor[:, 0].eval(session=tf.compat.v1.Session()),
                                                   axis=0), path_length - 1, 0), (-1, embed_dim))
    for i in range(1, path_length):
        additional_to_repeat = tf.gather(node_embeddings,
                                         pos_tensor[:, i].eval(session=tf.compat.v1.Session()),
                                         axis=0)
        left_part_pos = tf.concat([left_part_pos,
                                   tf.reshape(tf.repeat(additional_to_repeat, path_length - 1 - i, 0),
                                              (-1, embed_dim))
                                   ],
                                  axis=0)

    right_part_unstacked = tf.gather(node_embeddings,
                                     tf.reshape(pos_tensor[:, 1:], (-1,)).eval(session=tf.compat.v1.Session()),
                                     axis=0)
    right_part_pos = tf.concat([right_part_unstacked], axis=0)
    for i in range(1, path_length - 1):
        part_to_concat = tf.gather(node_embeddings,
                                   tf.reshape(pos_tensor[:, 1 + i:], (-1,)).eval(session=tf.compat.v1.Session()),
                                   axis=0)
        right_part_pos = tf.concat([right_part_pos] + [part_to_concat], axis=0)

    out_p = tf.math.log_sigmoid(
        tf.matmul(tf.reshape(left_part_pos, (int(path_length * (path_length - 1) / 2) * len(pos_instances), 1, -1)),
                  tf.reshape(right_part_pos, (int(path_length * (path_length - 1) / 2) * len(pos_instances), -1, 1))))
    # ----- putting the contributions together and returning
    # tensorflow-pytorch comparison prints
    # print(out_p.eval(session=tf.compat.v1.Session()).reshape(-1))
    # print(out_n.eval(session=tf.compat.v1.Session()).reshape(-1))

    return tf.reduce_mean(-out_p) - tf.reduce_mean(out_n)


def NSHE_network_schema_loss(predict, ns_label):
    BCE_loss = torch.nn.BCELoss()
    return BCE_loss(predict, ns_label)
