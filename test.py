import pandas as pd
import torch
from torch_geometric.utils import structured_negative_sampling
from datasets import DBLP_MAGNN
from models import RGCN
from data_transforms import dblp_for_rgcn
from utils.losses import triplet_loss_pure, triplet_loss_type_aware
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
test = DBLP_MAGNN(root="/home/ubuntu/msandal_code/PyG_playground/data/dblp", use_MAGNN_init_feats=True)
node_type_id_dict, edge_type_id_dict, id_type_mask, node_feature_matrix, edge_index, edge_type = dblp_for_rgcn()
print('Data transformed!')

model = RGCN(input_dim=50, output_dim=30, hidden_dim=40, num_relations=2)
optimizer = torch.optim.AdamW(model.parameters())
n_epochs = 200
losses = []

for epoch in range(n_epochs):
    model = model.float()
    output = model(x=node_feature_matrix.float(),
                   edge_index=edge_index,
                   edge_type=edge_type)
    triplets = structured_negative_sampling(edge_index)
    loss = triplet_loss_type_aware(triplets, output, id_type_mask, lmbd=10)
    losses.append(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print("Epoch: ", epoch, " loss: ", loss)

writer.add_embedding(output,
                     metadata=id_type_mask)
writer.flush()
writer.close()
