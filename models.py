import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import RGCNConv
from typing import Union, Tuple
from torch_geometric.typing import OptTensor, Adj


class RGCN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int,
                 num_relations: int, num_layers: int = 3, sigma: str = 'relu'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_relations = num_relations
        self.activation = F.relu if sigma == 'relu' else None
        if self.activation is None:
            raise NotImplementedError('Currently only relu activation supported')
        super(RGCN, self).__init__()

        self.gcs = nn.ModuleList()
        for i in range(self.num_layers):
            in_channels = self.input_dim if i == 0 else self.hidden_dim
            out_channels = self.hidden_dim if i != self.num_layers - 1 else self.output_dim
            self.gcs.append(RGCNConv(in_channels=in_channels,
                                     out_channels=out_channels,
                                     num_relations=self.num_relations))

    def forward(self, x: Union[OptTensor, Tuple[OptTensor, torch.Tensor]],
                edge_index: Adj, edge_type: OptTensor = None):
        """
        For a good reference on the required parameters please refer to the official
        torch_geometric documentation for RGCNConv:
        https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/rgcn_conv.html#RGCNConv
        """
        x_hat = x
        for gc in self.gcs:
            x_hat = self.activation(gc(x=x_hat, edge_index=edge_index, edge_type=edge_type))
        return x_hat
