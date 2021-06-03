import torch
import math
import torch.nn as nn
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
import torch.nn.functional as F
import numpy as np

# ###############################################
#         Graph Transformer Network (GTN) conv
# ###############################################
class GTConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 1, 1))
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, 0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        A = torch.sum(A * F.softmax(self.weight, dim=1), dim=1)
        return A


class GTLayer(nn.Module):

    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        if self.first:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)

    def forward(self, A, H_=None):
        if self.first:
            a = self.conv1(A)
            b = self.conv2(A)
            H = torch.bmm(a, b)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach(), (F.softmax(self.conv2.weight, dim=1)).detach()]
        else:
            a = self.conv1(A)
            H = torch.bmm(H_, a)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        return H, W


# ###############################################
# NSHE convs: code from https://github.com/Andy-Border/NSHE
# ###############################################

class GraphConvolution(Module):  # GCN AHW
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj, global_W=None):
        if len(adj._values()) == 0:
            return torch.zeros(adj.shape[0], self.out_features)
        support = torch.spmm(inputs, self.weight)  # HW in GCN
        if global_W is not None:
            support = torch.spmm(support, global_W)  # Ignore this!
        output = torch.spmm(adj, support)  # AHW
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphAttentionConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphAttentionConvolution, self).__init__()
        self.out_dim = out_features
        self.weights = Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_normal_(self.weights.data, gain=1.414)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            stdv = 1. / math.sqrt(out_features)
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)
        self.attention = Attention_InfLevel(out_features)

    def forward(self, input_, adj, global_W=None):

        h = torch.spmm(input_, self.weights)
        h_prime = self.attention(h, adj) + self.bias
        return h_prime


class Attention_InfLevel(nn.Module):
    def __init__(self, dim_features):
        super(Attention_InfLevel, self).__init__()
        self.dim_features = dim_features
        self.a1 = nn.Parameter(torch.zeros(size=(dim_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(dim_features, 1)))
        nn.init.xavier_normal_(self.a1.data, gain=1.414)
        nn.init.xavier_normal_(self.a2.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, h, adj):
        N = h.size()[0]
        e1 = torch.matmul(h, self.a1).repeat(1, N)
        e2 = torch.matmul(h, self.a2).repeat(1, N).t()
        e = e1 + e2
        e = self.leakyrelu(e)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.to_dense() > 0, e, zero_vec)
        del zero_vec
        attention = F.softmax(attention, dim=1)
        h_prime = torch.matmul(attention, h)  # h' = alpha * h(hw)
        return h_prime


# ###############################################
# MAGNN convs: code from https://github.com/yangji9181/HNE
# ###############################################

class MAGNN_mptype_layer(nn.Module):

    def __init__(self, etypes, odim, device, nhead, dropout, rvec, rtype='RotatE0', alpha=0.01):
        super(MAGNN_mptype_layer, self).__init__()

        self.etypes = etypes
        self.odim = odim
        self.nhead = nhead
        self.rvec = rvec
        self.rtype = rtype

        # rnn-like metapath instance aggregator
        # consider multiple attention heads
        if rtype == 'gru':
            self.rnn = nn.GRU(odim, nhead * odim).to(device)
        elif rtype == 'lstm':
            self.rnn = nn.LSTM(odim, nhead * odim).to(device)
        elif rtype == 'bi-gru':
            self.rnn = nn.GRU(odim, nhead * odim // 2, bidirectional=True).to(device)
        elif rtype == 'bi-lstm':
            self.rnn = nn.LSTM(odim, nhead * odim // 2, bidirectional=True).to(device)
        elif rtype == 'linear':
            self.rnn = nn.Linear(odim, nhead * odim).to(device)
        elif rtype == 'max-pooling':
            self.rnn = nn.Linear(odim, nhead * odim).to(device)
        elif rtype == 'neighbor-linear':
            self.rnn = nn.Linear(odim, nhead * odim).to(device)

        # node-level attention
        # attention considers the center node embedding
        self.attn1 = nn.Linear(odim, nhead, bias=False).to(device)
        self.attn2 = nn.Parameter(torch.empty(size=(1, nhead, odim))).to(device)
        nn.init.xavier_normal_(self.attn1.weight, gain=1.414)
        nn.init.xavier_normal_(self.attn2.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = edge_softmax
        self.attn_drop = nn.Dropout(dropout) if dropout > 0 else lambda x: x

    def edge_softmax(self, g):
        attention = self.softmax(g, g.edata.pop('a'))
        g.edata['a_drop'] = self.attn_drop(attention)

    def message_passing(self, edges):
        ft = edges.data['eft'] * edges.data['a_drop']
        return {'ft': ft}

    def forward(self, g, mpinstances, iftargets, input_node_features):

        edata = []
        for mpinstance in mpinstances:
            edata.append(torch.stack([input_node_features[node] for node in mpinstance]))
        edata = torch.stack(edata)
        center_node_feat = torch.clone(edata[:, -1, :])

        # apply rnn to metapath-based feature sequence
        if self.rtype == 'gru':
            _, hidden = self.rnn(edata.permute(1, 0, 2))
        elif self.rtype == 'lstm':
            _, (hidden, _) = self.rnn(edata.permute(1, 0, 2))
        elif self.rtype == 'bi-gru':
            _, hidden = self.rnn(edata.permute(1, 0, 2))
            hidden = hidden.permute(1, 0, 2).reshape(-1, self.odim, self.nhead).permute(0, 2, 1).reshape(
                -1, self.nhead * self.odim).unsqueeze(dim=0)
        elif self.rtype == 'bi-lstm':
            _, (hidden, _) = self.rnn(edata.permute(1, 0, 2))
            hidden = hidden.permute(1, 0, 2).reshape(-1, self.odim, self.nhead).permute(0, 2, 1).reshape(
                -1, self.nhead * self.odim).unsqueeze(dim=0)
        elif self.rtype == 'average':
            hidden = torch.mean(edata, dim=1)
            hidden = torch.cat([hidden] * self.nhead, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rtype == 'linear':
            hidden = self.rnn(torch.mean(edata, dim=1))
            hidden = hidden.unsqueeze(dim=0)
        elif self.rtype == 'max-pooling':
            hidden, _ = torch.max(self.rnn(edata), dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rtype == 'TransE0' or self.rtype == 'TransE1':
            rvec = self.rvec
            if self.rtype == 'TransE0':
                rvec = torch.stack((rvec, -rvec), dim=1)
                rvec = rvec.reshape(self.rvec.shape[0] * 2, self.rvec.shape[1])  # etypes x odim
            edata = F.normalize(edata, p=2, dim=2)
            for i in range(edata.shape[1] - 1):
                # consider None edge (symmetric relation)
                temp_etypes = [etype for etype in self.etypes[i:] if etype is not None]
                edata[:, i] = edata[:, i] + rvec[temp_etypes].sum(dim=0)
            hidden = torch.mean(edata, dim=1)
            hidden = torch.cat([hidden] * self.nhead, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rtype == 'RotatE0' or self.rtype == 'RotatE1':
            rvec = F.normalize(self.rvec, p=2, dim=2)
            if self.rtype == 'RotatE0':
                rvec = torch.stack((rvec, rvec), dim=1)
                rvec[:, 1, :, 1] = -rvec[:, 1, :, 1]
                rvec = rvec.reshape(self.rvec.shape[0] * 2, self.rvec.shape[1], 2)  # etypes x odim/2 x 2
            edata = edata.reshape(edata.shape[0], edata.shape[1], edata.shape[2] // 2, 2)
            final_rvec = torch.zeros([edata.shape[1], self.odim // 2, 2], device=edata.device)
            final_rvec[-1, :, 0] = 1
            for i in range(final_rvec.shape[0] - 2, -1, -1):
                # consider None edge (symmetric relation)
                if self.etypes[i] is not None:
                    final_rvec[i, :, 0] = final_rvec[i + 1, :, 0].clone() * rvec[self.etypes[i], :, 0] - \
                                          final_rvec[i + 1, :, 1].clone() * rvec[self.etypes[i], :, 1]
                    final_rvec[i, :, 1] = final_rvec[i + 1, :, 0].clone() * rvec[self.etypes[i], :, 1] + \
                                          final_rvec[i + 1, :, 1].clone() * rvec[self.etypes[i], :, 0]
                else:
                    final_rvec[i, :, 0] = final_rvec[i + 1, :, 0].clone()
                    final_rvec[i, :, 1] = final_rvec[i + 1, :, 1].clone()
            for i in range(edata.shape[1] - 1):
                temp1 = edata[:, i, :, 0].clone() * final_rvec[i, :, 0] - \
                        edata[:, i, :, 1].clone() * final_rvec[i, :, 1]
                temp2 = edata[:, i, :, 0].clone() * final_rvec[i, :, 1] + \
                        edata[:, i, :, 1].clone() * final_rvec[i, :, 0]
                edata[:, i, :, 0] = temp1
                edata[:, i, :, 1] = temp2
            edata = edata.reshape(edata.shape[0], edata.shape[1], -1)
            hidden = torch.mean(edata, dim=1)
            hidden = torch.cat([hidden] * self.nhead, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rtype == 'neighbor':
            hidden = edata[:, 0]
            hidden = torch.cat([hidden] * self.nhead, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rtype == 'neighbor-linear':
            hidden = self.rnn(edata[:, 0])
            hidden = hidden.unsqueeze(dim=0)

        eft = hidden.permute(1, 0, 2).view(-1, self.nhead, self.odim)

        a1 = self.attn1(center_node_feat)
        a2 = (eft * self.attn2).sum(dim=-1)
        a = (a1 + a2).unsqueeze(dim=-1)
        a = self.leaky_relu(a)
        g.edata.update({'eft': eft, 'a': a})

        # compute softmax normalized attention values
        self.edge_softmax(g)
        # compute the aggregated node features scaled by the dropped,
        # unnormalized attention values.
        g.update_all(self.message_passing, fn.sum('ft', 'ft'))

        targets = np.where(iftargets[:, 1] == 1)[0]
        target_features = g.ndata['ft'][targets]

        return iftargets[targets, 0], target_features


class MAGNN_ntype_layer(nn.Module):

    def __init__(self, mptype_etypes, odim, adim, device, nhead, dropout, rvec, rtype='RotatE0'):
        super(MAGNN_ntype_layer, self).__init__()

        self.odim = odim
        self.nhead = nhead

        # metapath-specific layers
        self.MAGNN_mptype_layers = {}
        for mptype, etypes in mptype_etypes.items():
            self.MAGNN_mptype_layers[mptype] = MAGNN_mptype_layer(etypes, odim, device, nhead, dropout, rvec, rtype)

        # metapath-level attention
        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        self.fc1 = nn.Linear(odim * nhead, adim, bias=True).to(device)
        self.fc2 = nn.Linear(adim, 1, bias=False).to(device)
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, mptype_g, mptype_mpinstances, mptype_iftargets, input_node_features):

        output_node_features = []
        for mptype in mptype_iftargets:
            targets, output_mptype_node_features = self.MAGNN_mptype_layers[mptype](mptype_g[mptype],
                                                                                    mptype_mpinstances[mptype],
                                                                                    mptype_iftargets[mptype],
                                                                                    input_node_features)
            output_node_features.append(F.elu(output_mptype_node_features).view(-1, self.odim * self.nhead))

        beta = []
        for each in output_node_features:
            fc1 = torch.tanh(self.fc1(each))
            fc2 = self.fc2(torch.mean(fc1, dim=0))
            beta.append(fc2)
        beta = F.softmax(torch.cat(beta, dim=0), dim=0)
        beta = torch.unsqueeze(torch.unsqueeze(beta, dim=-1), dim=-1)

        output_node_features = torch.cat([torch.unsqueeze(each, dim=0) for each in output_node_features], dim=0)
        output_node_features = torch.sum(beta * output_node_features, dim=0)

        return targets, output_node_features


class MAGNN_layer(nn.Module):

    def __init__(self, graph_statistics, idim, odim, adim, device, nhead, dropout, rtype='RotatE0'):
        super(MAGNN_layer, self).__init__()

        # etype-specific parameters
        rvec = None
        if rtype == 'TransE0':
            rvec = nn.Parameter(torch.empty(size=(graph_statistics['num_etype'] // 2, idim))).to(device)
        elif rtype == 'TransE1':
            rvec = nn.Parameter(torch.empty(size=(graph_statistics['num_etype'], idim))).to(device)
        elif rtype == 'RotatE0':
            rvec = nn.Parameter(torch.empty(size=(graph_statistics['num_etype'] // 2, idim // 2, 2))).to(device)
        elif rtype == 'RotatE1':
            rvec = nn.Parameter(torch.empty(size=(graph_statistics['num_etype'], idim // 2, 2))).to(device)
        if rvec is not None:
            nn.init.xavier_normal_(rvec.data, gain=1.414)

        # ntype-specific layer
        self.MAGNN_ntype_layers = {}
        for ntype, mptype_etypes in graph_statistics['ntype_mptype_etypes'].items():
            self.MAGNN_ntype_layers[ntype] = MAGNN_ntype_layer(mptype_etypes, idim, adim, device, nhead, dropout, rvec,
                                                               rtype)

        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        self.fc = nn.Linear(idim * nhead, odim, bias=True).to(device)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

    def forward(self, ntype_mptype_g, ntype_mptype_mpinstances, ntype_mptype_iftargets, input_node_features):

        # ntype-specific layer
        ntype_targets, ntype_output_node_features = [], []
        for ntype in ntype_mptype_g:
            output_targets, output_node_features = self.MAGNN_ntype_layers[ntype](ntype_mptype_g[ntype],
                                                                                  ntype_mptype_mpinstances[ntype],
                                                                                  ntype_mptype_iftargets[ntype],
                                                                                  input_node_features)
            ntype_targets.append(output_targets)
            ntype_output_node_features.append(output_node_features)

        targets = np.concatenate(ntype_targets)
        transformed = F.elu(self.fc(torch.cat(ntype_output_node_features)))
        node_features = {node: feature for node, feature in zip(targets, transformed)}

        return node_features
