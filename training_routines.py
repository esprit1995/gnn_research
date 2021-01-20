import torch
import pandas as pd
from models import RGCN, GTN
from data_transforms import IMDB_ACM_DBLP_for_rgcn, IMDB_ACM_DBLP_for_gtn
from utils.tools import heterogeneous_negative_sampling_naive
from utils.losses import triplet_loss_pure, triplet_loss_type_aware


def train_rgcn(args):
    # RGCN settings ##########
    output_dim = 50
    hidden_dim = 50
    num_layers = 3
    # #######################
    n_node_dict, node_label_dict, id_type_mask, node_feature_matrix, edge_index, edge_type = IMDB_ACM_DBLP_for_rgcn(
        name=args.dataset)
    print('Data transformed!')

    model = RGCN(input_dim=node_feature_matrix.shape[1],
                 output_dim=output_dim,
                 hidden_dim=hidden_dim,
                 num_relations=pd.Series(edge_type.numpy()).nunique(),
                 num_layers=num_layers)
    optimizer = torch.optim.AdamW(model.parameters())
    losses = []
    for epoch in range(args.epochs):
        model = model.float()
        output = model(x=node_feature_matrix.float(),
                       edge_index=edge_index,
                       edge_type=edge_type)
        triplets = heterogeneous_negative_sampling_naive(edge_index, id_type_mask, random_seed=args.random_seed)
        loss = triplet_loss_type_aware(triplets, output, id_type_mask, lmbd=args.type_lambda) if args.type_aware_loss \
            else triplet_loss_pure(triplets, output)
        losses.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print("Epoch: ", epoch, " loss: ", loss)

    return output, id_type_mask


def train_gtn(args):
    # GTN settings ##########
    node_dim = 10
    num_channels = 2
    num_layers = 2
    norm = 'true'
    # #######################
    A, node_features, num_classes, edge_index, edge_type, id_type_mask = IMDB_ACM_DBLP_for_gtn(name='ACM')
    model = GTN(num_edge=A.shape[-1],
                num_channels=num_channels,
                w_in=node_features.shape[1],
                w_out=node_dim,
                num_class=num_classes,
                num_layers=num_layers,
                norm=norm)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
    for epoch in range(args.epochs):
        model.zero_grad()
        output = model(A, node_features)
        triplets = heterogeneous_negative_sampling_naive(edge_index, id_type_mask, random_seed=args.random_seed)
        loss = triplet_loss_type_aware(triplets, output, id_type_mask, lmbd=args.type_lambda) if args.type_aware_loss \
            else triplet_loss_pure(triplets, output)
        loss.backward()
        optimizer.step()
        print("Epoch: ", epoch, " loss: ", loss)

    return output, id_type_mask
