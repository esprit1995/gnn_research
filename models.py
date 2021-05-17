import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import RGCNConv
from typing import Union, Tuple
from torch_geometric.typing import OptTensor, Adj

from conv import GTLayer
from conv import GraphConvolution, GraphAttentionConvolution

# ###############################################
#   Relational Graph Convolution Network (RGCN)
# ###############################################

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
                                     num_relations=self.num_relations,
                                     num_bases=30))

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


# ###############################################
#         Graph Transformer Network (GTN)
# ###############################################
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


# ###############################################
#  Network Schema-preserving HIN Embedding (NSHE)
# ###############################################
# a little trick for layer lists
class AttrProxy(object):
    """Translates index lookups into attribute lookups."""

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class NS_MLP_Classifier(nn.Module):
    def __init__(self, in_feat, hidden_dim=[16]):
        super(NS_MLP_Classifier, self).__init__()
        self.hidden_layer = nn.Linear(in_feat, hidden_dim[0])
        self.output_layer = nn.Linear(hidden_dim[-1], 1)
        return

    def forward(self, input):
        ns_x = F.relu(self.hidden_layer(input))
        ns_y = self.output_layer(ns_x)
        ns_y = F.sigmoid(ns_y).flatten()
        return ns_y


class NSHE(nn.Module):

    def __init__(self, g, hp):
        super(NSHE, self).__init__()
        self.conv_method = hp.conv_method
        self.cla_layers = hp.cla_layers
        self.ns_emb_mode = hp.ns_emb_mode
        self.cla_method = hp.cla_method
        self.norm_emb = hp.norm_emb_flag
        self.types = g.node_types
        size = hp.size
        self.t_info = g.t_info
        for t in self.types:
            self.add_module('encoder_' + t, nn.Linear(g.feature[t].shape[1], size['com_feat_dim']))
        self.encoder = AttrProxy(self, 'encoder_')
        self.non_linear = nn.ReLU()
        self.context_dim = int(size['emb_dim'] / (len(self.types) - 1))
        # * ================== Neighborhood Agg==================
        emb_dim = size['emb_dim']
        if self.conv_method[:3] == 'GAT':
            self.neig_aggregator = GraphAttentionConvolution(size['com_feat_dim'], size['emb_dim'])
            if self.conv_method[-1] == '2':
                emb_dim = int(size['emb_dim'] / 2)
                self.neig_aggregator_2 = GraphAttentionConvolution(size['emb_dim'], emb_dim)
        elif self.conv_method[:3] == 'GCN':
            self.neig_aggregator = GraphConvolution(size['com_feat_dim'], size['emb_dim'])
            if self.conv_method[-1] == '2':
                emb_dim = int(size['emb_dim'] / 2)
                self.neig_aggregator_2 = GraphConvolution(size['emb_dim'], emb_dim)
        # * ================== NSI Embedding Gen=================
        if self.cla_method == 'TypeSpecCla':
            for t in self.types:
                self.add_module('nsi_encoder' + t, nn.Linear(emb_dim, self.context_dim))
            self.nsi_encoder = AttrProxy(self, 'nsi_encoder')
        # * ================== NSI Classification================
        if self.cla_method == '2layer':
            if self.ns_emb_mode == 'TypeLvAtt':
                self.ns_classifier = NS_MLP_Classifier(
                    emb_dim, [int(emb_dim / 2)])
            elif self.ns_emb_mode == 'Concat':
                self.ns_classifier = NS_MLP_Classifier(len(g.t_info) * emb_dim, emb_dim)
        elif self.cla_method == 'TypeSpecCla':
            for t in self.types:
                if self.cla_layers == 1:
                    self.add_module('ns_cla_' + t, nn.Linear(emb_dim + self.context_dim * (len(self.types) - 1), 1))
                else:
                    self.add_module('ns_cla_' + t,
                                    NS_MLP_Classifier(emb_dim + self.context_dim * (len(self.types) - 1), [16]))
            self.ns_classifier = AttrProxy(self, 'ns_cla_')
        print(self)

    def forward(self, adj, features, nsi_list):
        # * =============== Encode heterogeneous feature ================
        #
        encoded = torch.cat([self.non_linear(self.encoder[t](features[t])) for t in self.types])
        # * =============== Node Embedding Generation ===================
        com_emb = self.neig_aggregator(encoded, adj)
        if self.conv_method[-1] == '2':
            com_emb = self.neig_aggregator_2(com_emb, adj)
        if self.norm_emb:
            # Independently normalize each dimension
            com_emb = F.normalize(com_emb, p=2, dim=1)

        return com_emb
