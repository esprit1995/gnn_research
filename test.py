import pandas as pd
import torch
from models import RGCN
from data_transforms import IMDB_ACM_DBLP_for_rgcn
from utils.tools import heterogeneous_negative_sampling_naive
from utils.losses import triplet_loss_pure, triplet_loss_type_aware
from utils.arguments import model_run_argparse
from torch.utils.tensorboard import SummaryWriter

# edge_index = torch.tensor([[0, 1, 1, 1, 2, 2, 3, 3], [4, 5, 6, 8, 7, 9, 6, 10]])
# node_idx_type = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
# print(hetergeneous_negative_sampling_naive(edge_index, node_idx_type, random_seed=1))

experiment_name = 'test_argparse_run'
args = model_run_argparse()
writer = SummaryWriter("runs/" + str(experiment_name))
print(args)
if args.model == 'RGCN':
    n_node_dict, node_label_dict, id_type_mask, node_feature_matrix, edge_index, edge_type = IMDB_ACM_DBLP_for_rgcn(
        name=args.dataset)
    print('Data transformed!')

    model = RGCN(input_dim=node_feature_matrix.shape[1],
                 output_dim=args.output_dim,
                 hidden_dim=args.hidden_dim,
                 num_relations=pd.Series(edge_type.numpy()).nunique())
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

    writer.add_embedding(output, metadata=id_type_mask)
    writer.flush()
    writer.close()
else:
    raise NotImplementedError("No implementation for model name: ", args.model)
