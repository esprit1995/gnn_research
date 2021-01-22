import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import RGCNConv
from typing import Union, Tuple
from torch_geometric.typing import OptTensor, Adj

from conv import GTLayer
from conv import HANLayer


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


class GTN(nn.Module):

    def __init__(self, num_edge, num_channels, w_in, w_out, num_class, num_layers, norm):
        super(GTN, self).__init__()
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(num_edge, num_channels, first=True))
            else:
                layers.append(GTLayer(num_edge, num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        self.weight = nn.Parameter(torch.Tensor(w_in, w_out))
        self.bias = nn.Parameter(torch.Tensor(w_out))
        # self.loss = nn.CrossEntropyLoss()
        self.linear1 = nn.Linear(self.w_out * self.num_channels, self.w_out)
        self.linear2 = nn.Linear(self.w_out, self.num_class)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def gcn_conv(self, X, H):
        X = torch.mm(X, self.weight)
        H = self.norm(H, add=True)
        return torch.mm(H.t(), X)

    def normalization(self, H):
        for i in range(self.num_channels):
            if i == 0:
                H_ = self.norm(H[i, :, :]).unsqueeze(0)
            else:
                H_ = torch.cat((H_, self.norm(H[i, :, :]).unsqueeze(0)), dim=0)
        return H_

    def norm(self, H, add=False):
        H = H.t()
        if not add:
            H = H * ((torch.eye(H.shape[0]) == 0).type(torch.FloatTensor))
        else:
            H = H * ((torch.eye(H.shape[0]) == 0).type(torch.FloatTensor)) + torch.eye(H.shape[0]).type(
                torch.FloatTensor)
        deg = torch.sum(H, dim=1)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        deg_inv = deg_inv * torch.eye(H.shape[0]).type(torch.FloatTensor)
        H = torch.mm(deg_inv, H)
        H = H.t()
        return H

    def forward(self, A, X):  # , target_x, target):
        A = A.unsqueeze(0).permute(0, 3, 1, 2)
        Ws = []
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)
            else:
                H = self.normalization(H)
                H, W = self.layers[i](A, H)
            Ws.append(W)

        # H,W1 = self.layer1(A)
        # H = self.normalization(H)
        # H,W2 = self.layer2(A, H)
        # H = self.normalization(H)
        # H,W3 = self.layer3(A, H)
        for i in range(self.num_channels):
            if i == 0:
                X_ = F.relu(self.gcn_conv(X, H[i]))
            else:
                X_tmp = F.relu(self.gcn_conv(X, H[i]))
                X_ = torch.cat((X_, X_tmp), dim=1)
        X_ = self.linear1(X_)
        # X_ = F.relu(X_)
        # y = self.linear2(X_[target_x])
        # loss = self.loss(y, target)
        return X_


class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[l - 1],
                                        hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h)
