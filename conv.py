import torch
import math
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F


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
