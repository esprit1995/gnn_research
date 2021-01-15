import torch
import torch.nn as nn
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
        "triplet_loss_pure(): triplet lists of unequal length!"

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
    type_negative = torch.cat((node_embeddings[id_triplets[1][np.logical_not(type_comparison_mask)]].view(-1, emb_dim, 1),
                               node_embeddings[id_triplets[2][np.logical_not(type_comparison_mask)]].view(-1, emb_dim, 1)), 0)
    out_p = F.logsigmoid(torch.bmm(torch.cat(2*[central_positive],0), type_positive))
    out_n = F.logsigmoid(- torch.bmm(torch.cat(2*[central_negative],0), type_negative))
    type_loss = - (out_p.mean() if out_p.numel() != 0 else 0) - (out_n.mean() if out_n.numel() != 0 else 0)
    return normal_loss + lmbd*type_loss
