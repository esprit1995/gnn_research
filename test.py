import pandas as pd
import torch
from datasets import DBLP_MAGNN
from models import RGCN
from data_transforms import dblp_for_rgcn
from utils.tools import hetergeneous_negative_sampling_naive
from utils.losses import triplet_loss_pure, triplet_loss_type_aware
from torch.utils.tensorboard import SummaryWriter

# edge_index = torch.tensor([[0, 1, 1, 1, 2, 2, 3, 3], [4, 5, 6, 8, 7, 9, 6, 10]])
# node_idx_type = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
# print(hetergeneous_negative_sampling_naive(edge_index, node_idx_type, random_seed=1))

experiment_name = '500epoch05lmbd_new_sampling_new_typeloss'

writer = SummaryWriter("runs/"+str(experiment_name))
test = DBLP_MAGNN(root="/home/ubuntu/msandal_code/PyG_playground/data/dblp", use_MAGNN_init_feats=True)
node_type_id_dict, n_node_dict, node_label_dict,\
    edge_type_id_dict, id_type_mask, node_feature_matrix, edge_index, edge_type = dblp_for_rgcn()
print('Data transformed!')

model = RGCN(input_dim=50, output_dim=30, hidden_dim=40, num_relations=2)
optimizer = torch.optim.AdamW(model.parameters())
n_epochs = 500
losses = []

for epoch in range(n_epochs):
    model = model.float()
    output = model(x=node_feature_matrix.float(),
                   edge_index=edge_index,
                   edge_type=edge_type)
    triplets = hetergeneous_negative_sampling_naive(edge_index, id_type_mask, random_seed=10)
    loss = triplet_loss_type_aware(triplets, output, id_type_mask, lmbd=0.5)
    losses.append(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print("Epoch: ", epoch, " loss: ", loss)

labels = node_label_dict['author']

writer.add_embedding(output[:-n_node_dict['term']-n_node_dict['paper']],
                     metadata=labels)
writer.flush()
writer.close()
