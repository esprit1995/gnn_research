import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


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


def push_pull_metapath_instance_loss(pos_instance: list, neg_instance: list,
                                     node_embeddings: torch.tensor):
    """
    compute the push-pull loss (logsigmoid loss) over the given metapath instances.
    assumes that positive and negative metapath instances have the same template
    :param pos_instance: list of tuples containing positive metapath instances
    :param neg_instance: list of typles containing negative metapath instances
    :param node_embeddings: current embeddings of the nodes
    :return: push-pull loss
    """
    path_length = len(pos_instance[0])
    pos_tensor = torch.tensor(pos_instance)
    neg_tensor = torch.tensor(neg_instance)
    # computing individual dot products
    left_part_pos = torch.vstack((path_length - 1) * [node_embeddings[pos_tensor[:, 0]]])
    left_part_neg = torch.vstack((path_length - 1) * [node_embeddings[neg_tensor[:, 0]]])
    for i in range(1, path_length):
        left_part_pos = torch.vstack([left_part_pos] + (path_length - 1 - i) * [node_embeddings[pos_tensor[:, i]]])
        left_part_neg = torch.vstack([left_part_neg] + (path_length - 1 - i) * [node_embeddings[neg_tensor[:, i]]])

    right_part_pos = torch.vstack([node_embeddings[pos_tensor[:, 1:].reshape(-1)]])
    right_part_neg = torch.vstack([node_embeddings[neg_tensor[:, 1:].reshape(-1)]])
    for i in range(1, path_length - 1):
        right_part_pos = torch.vstack([right_part_pos] + [node_embeddings[pos_tensor[:, 1 + i:].reshape(-1)]])
        right_part_neg = torch.vstack([right_part_neg] + [node_embeddings[neg_tensor[:, 1 + i:].reshape(-1)]])

    out_p = F.logsigmoid(torch.bmm(left_part_pos.view(int(path_length * (path_length - 1) / 2) * len(pos_instance), 1, -1),
                                   right_part_pos.view(int(path_length * (path_length - 1) / 2) * len(pos_instance), -1, 1)))
    out_n = F.logsigmoid(-torch.bmm(left_part_neg.view(int(path_length * (path_length - 1) / 2) * len(neg_instance), 1, -1),
                                    right_part_neg.view(int(path_length * (path_length - 1) / 2) * len(neg_instance), -1, 1)))
    return -out_p.mean() - out_n.mean()
