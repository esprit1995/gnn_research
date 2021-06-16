import os
from tqdm import tqdm
from typing import Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.HeGAN_utils as HeGAN_utils

from downstream_tasks.evaluation_funcs import evaluate_clu_cla_GTN_NSHE_datasets

from torch_geometric.nn.conv import RGCNConv
from torch_geometric.typing import OptTensor, Adj

from utils.losses import push_pull_metapath_instance_loss_tf

from conv import GTLayer
from conv import GraphConvolution, GraphAttentionConvolution
from conv import MAGNN_layer

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf


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


# ###############################################
#  Metapath Aggregating Graph Neural Network (MAGNN)
# ###############################################
class MAGNN(nn.Module):

    # graph_statistics: num_etype (hetero edges only), {ntype: idim}, {ntype: mptype: etypes (homo edges are typed None)}
    def __init__(self, graph_statistics, hdim, adim, dropout, device, nlayer, nhead, nlabel=0, ntype_features={},
                 rtype='RotatE0'):
        super(MAGNN, self).__init__()

        self.device = device
        self.attributed = False if len(ntype_features) == 0 else True
        self.supervised = False if nlabel == 0 else True

        # ntype-specific transformation
        self.ntype_transformation = {}
        for ntype, idim in graph_statistics['ntype_idim'].items():
            if self.attributed:
                self.ntype_transformation[ntype] = (
                    nn.Embedding.from_pretrained(torch.from_numpy(ntype_features[ntype])).to(self.device),
                    nn.Linear(idim, hdim, bias=True).to(self.device))
                nn.init.xavier_normal_(self.ntype_transformation[ntype][1].weight, gain=1.414)
            else:
                self.ntype_transformation[ntype] = nn.Embedding(idim, hdim).to(self.device)
        self.feat_drop = nn.Dropout(dropout) if dropout > 0 else lambda x: x

        # MAGNN layers
        self.MAGNN_layers = nn.ModuleList()
        for l in range(nlayer):
            self.MAGNN_layers.append(MAGNN_layer(graph_statistics, hdim, hdim, adim, device, nhead, dropout, rtype))

        # prediction layer
        if self.supervised:
            self.final = nn.Linear(hdim, nlabel, bias=True).to(self.device)
            nn.init.xavier_normal_(self.final.weight, gain=1.414)

    def forward(self, layer_ntype_mptype_g, layer_ntype_mptype_mpinstances, layer_ntype_mptype_iftargets,
                batch_ntype_orders):

        # ntype-specific transformation
        node_features = {}
        for ntype, node_orders in batch_ntype_orders.items():
            inputs = torch.from_numpy(np.array(list(node_orders.values())).astype(np.int64)).to(self.device)
            if self.attributed:
                transformed = self.ntype_transformation[ntype][1](self.ntype_transformation[ntype][0](inputs))
            else:
                transformed = self.ntype_transformation[ntype](inputs)
            transformed = self.feat_drop(transformed)
            node_features.update({node: feature for node, feature in zip(node_orders, transformed)})

        # MAGNN layers
        for l, layer in enumerate(self.MAGNN_layers):
            node_features = layer(layer_ntype_mptype_g[l], layer_ntype_mptype_mpinstances[l],
                                  layer_ntype_mptype_iftargets[l], node_features)

        node_preds = {}
        if self.supervised:
            inputs = torch.stack(list(node_features.values()))
            preds = self.final(inputs)
            node_preds.update({node: pred for node, pred in zip(node_features, preds)})

        if self.device == 'cuda': torch.cuda.empty_cache()

        return node_features, node_preds


# ###############################################
#  HIN embedding with GAN-based adv. learning (HeGAN)
# ###############################################

class Generator:
    def __init__(self, n_node, n_relation, node_emd_init, relation_emd_init, config, hidden_dim=-1):
        self.n_node = n_node
        self.n_relation = n_relation
        self.node_emd_init = node_emd_init
        self.relation_emd_init = relation_emd_init
        self.emd_dim = node_emd_init.shape[1]
        self.hidden_dim = self.emd_dim if hidden_dim == -1 else hidden_dim
        self.node_embedding_matrix_raw = tf.get_variable(name="gen_node_embedding_raw",
                                                         shape=self.node_emd_init.shape,
                                                         initializer=tf.constant_initializer(self.node_emd_init),
                                                         trainable=True)
        self.gen_transf_W = tf.get_variable(name='gen_transf_W',
                                            shape=[self.emd_dim, self.hidden_dim],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                            trainable=True)
        self.gen_transf_b = tf.get_variable(name='gen_transf_b',
                                            shape=[self.hidden_dim],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                            trainable=True)
        self.node_embedding_matrix = tf.get_variable(name="gen_node_embedding",
                                                     shape=self.node_emd_init.shape,
                                                     trainable=True)
        self.node_embedding_matrix = tf.matmul(self.node_embedding_matrix_raw, self.gen_transf_W) + self.gen_transf_b
        self.relation_embedding_matrix = tf.get_variable(name="gen_relation_embedding",
                                                         shape=[self.n_relation, self.hidden_dim, self.hidden_dim],
                                                         initializer=tf.contrib.layers.xavier_initializer(
                                                             uniform=False),
                                                         trainable=True)

        self.gen_w_1 = tf.get_variable(name='gen_w',
                                       shape=[self.hidden_dim, self.hidden_dim],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                       trainable=True)
        self.gen_b_1 = tf.get_variable(name='gen_b',
                                       shape=[self.hidden_dim],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                       trainable=True)
        self.gen_w_2 = tf.get_variable(name='gen_w_2',
                                       shape=[self.hidden_dim, self.hidden_dim],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                       trainable=True)
        self.gen_b_2 = tf.get_variable(name='gen_b_2',
                                       shape=[self.hidden_dim],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                       trainable=True)

        self.node_id = tf.placeholder(tf.int32, shape=[None])
        self.relation_id = tf.placeholder(tf.int32, shape=[None])
        self.noise_embedding = tf.placeholder(tf.float32, shape=[None, self.hidden_dim])

        self.dis_node_embedding = tf.placeholder(tf.float32, shape=[None, self.hidden_dim])
        self.dis_relation_embedding = tf.placeholder(tf.float32, shape=[None, self.hidden_dim, self.hidden_dim])

        self.node_embedding = tf.nn.embedding_lookup(self.node_embedding_matrix, self.node_id)
        self.relation_embedding = tf.nn.embedding_lookup(self.relation_embedding_matrix, self.relation_id)
        self.node_neighbor_embedding = self.generate_node(self.node_embedding, self.relation_embedding,
                                                          self.noise_embedding)

        t = tf.reshape(tf.matmul(tf.expand_dims(self.dis_node_embedding, 1), self.dis_relation_embedding),
                       [-1, self.hidden_dim])
        self.score = tf.reduce_sum(tf.multiply(t, self.node_neighbor_embedding), axis=1)

        self.loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.score), logits=self.score)) \
                    + config.lambda_gen * (tf.nn.l2_loss(self.node_embedding) + tf.nn.l2_loss(
            self.relation_embedding) + tf.nn.l2_loss(self.gen_w_1))

        optimizer = tf.train.AdamOptimizer(config.lr_gen)
        self.g_updates = optimizer.minimize(self.loss)

    def generate_node(self, node_embedding, relation_embedding, noise_embedding):
        # node_embedding = tf.nn.embedding_lookup(self.node_embedding_matrix, node_id)
        # relation_embedding = tf.nn.embedding_lookup(self.relation_embedding_matrix, relation_id)

        input = tf.reshape(tf.matmul(tf.expand_dims(node_embedding, 1), relation_embedding), [-1, self.hidden_dim])
        # input = tf.concat([input, noise_embedding], axis = 1)
        input = input + noise_embedding

        output = tf.nn.leaky_relu(tf.matmul(input, self.gen_w_1) + self.gen_b_1)
        # input = tf.nn.leaky_relu(tf.matmul(input, self.gen_w_1) + self.gen_b_1)# +  relation_embedding
        # output = tf.nn.leaky_relu(tf.matmul(input, self.gen_w_2) + self.gen_b_2)
        # output = node_embedding + relation_embedding + noise_embedding

        return output


class Discriminator:
    def __init__(self, n_node, n_relation, node_emd_init, relation_emd_init, config,
                 hidden_dim=-1, pos_instances=None, neg_instances=None, corruption_pos=None):
        self.n_node = n_node
        self.n_relation = n_relation
        self.node_emd_init = node_emd_init
        self.relation_emd_init = relation_emd_init
        self.emd_dim = node_emd_init.shape[1]
        self.hidden_dim = self.emd_dim if hidden_dim == -1 else hidden_dim

        # with tf.variable_scope('discriminator'):
        self.dis_w_1 = tf.get_variable(name='dis_w',
                                       shape=[self.emd_dim, self.hidden_dim],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                       trainable=True)
        self.dis_b_1 = tf.get_variable(name='dis_b',
                                       shape=[self.hidden_dim],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                       trainable=True)
        self.node_embedding_matrix_raw = tf.get_variable(name='dis_node_embedding_raw',
                                                         shape=self.node_emd_init.shape,
                                                         initializer=tf.constant_initializer(self.node_emd_init),
                                                         trainable=True)
        self.node_embedding_matrix = tf.get_variable(name='dis_node_embedding',
                                                     shape=self.node_emd_init.shape,
                                                     trainable=True)
        self.node_embedding_matrix = tf.matmul(self.node_embedding_matrix_raw, self.dis_w_1) + self.dis_b_1
        self.relation_embedding_matrix = tf.get_variable(name='dis_relation_embedding',
                                                         shape=[self.n_relation, self.hidden_dim, self.hidden_dim],
                                                         initializer=tf.contrib.layers.xavier_initializer(
                                                             uniform=False),
                                                         trainable=True)

        self.pos_node_id = tf.placeholder(tf.int32, shape=[None])
        self.pos_relation_id = tf.placeholder(tf.int32, shape=[None])
        self.pos_node_neighbor_id = tf.placeholder(tf.int32, shape=[None])

        self.neg_node_id_1 = tf.placeholder(tf.int32, shape=[None])
        self.neg_relation_id_1 = tf.placeholder(tf.int32, shape=[None])
        self.neg_node_neighbor_id_1 = tf.placeholder(tf.int32, shape=[None])

        self.neg_node_id_2 = tf.placeholder(tf.int32, shape=[None])
        self.neg_relation_id_2 = tf.placeholder(tf.int32, shape=[None])
        self.node_fake_neighbor_embedding = tf.placeholder(tf.float32, shape=[None, self.hidden_dim])

        self.pos_node_embedding = tf.nn.embedding_lookup(self.node_embedding_matrix, self.pos_node_id)
        self.pos_node_neighbor_embedding = tf.nn.embedding_lookup(self.node_embedding_matrix, self.pos_node_neighbor_id)
        self.pos_relation_embedding = tf.nn.embedding_lookup(self.relation_embedding_matrix, self.pos_relation_id)

        self.neg_node_embedding_1 = tf.nn.embedding_lookup(self.node_embedding_matrix, self.neg_node_id_1)
        self.neg_node_neighbor_embedding_1 = tf.nn.embedding_lookup(self.node_embedding_matrix,
                                                                    self.neg_node_neighbor_id_1)
        self.neg_relation_embedding_1 = tf.nn.embedding_lookup(self.relation_embedding_matrix, self.neg_relation_id_1)

        self.neg_node_embedding_2 = tf.nn.embedding_lookup(self.node_embedding_matrix, self.neg_node_id_2)
        self.neg_relation_embedding_2 = tf.nn.embedding_lookup(self.relation_embedding_matrix, self.neg_relation_id_2)

        # pos loss
        t = tf.reshape(tf.matmul(tf.expand_dims(self.pos_node_embedding, 1), self.pos_relation_embedding),
                       [-1, self.hidden_dim])
        self.pos_score = tf.reduce_sum(tf.multiply(t, self.pos_node_neighbor_embedding), axis=1)
        self.pos_loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.pos_score), logits=self.pos_score))

        # neg loss_1
        t = tf.reshape(tf.matmul(tf.expand_dims(self.neg_node_embedding_1, 1), self.neg_relation_embedding_1),
                       [-1, self.hidden_dim])
        self.neg_score_1 = tf.reduce_sum(tf.multiply(t, self.neg_node_neighbor_embedding_1), axis=1)
        self.neg_loss_1 = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.neg_score_1), logits=self.neg_score_1))

        # neg loss_2
        t = tf.reshape(tf.matmul(tf.expand_dims(self.neg_node_embedding_2, 1), self.neg_relation_embedding_2),
                       [-1, self.hidden_dim])
        self.neg_score_2 = tf.reduce_sum(tf.multiply(t, self.node_fake_neighbor_embedding), axis=1)
        self.neg_loss_2 = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.neg_score_2), logits=self.neg_score_2))

        # cocluster loss
        if neg_instances is None or pos_instances is None or corruption_pos is None:
            self.cocluster_loss = 0
            self.loss = self.pos_loss + self.neg_loss_1 + self.neg_loss_2
        else:
            mptemplates = list(pos_instances.keys())
            for idx in range(len(mptemplates)):
                self.cocluster_loss = config.cocluster_lambda * push_pull_metapath_instance_loss_tf(
                    pos_instances[mptemplates[idx]],
                    neg_instances[mptemplates[idx]],
                    corruption_pos[idx],
                    self.node_embedding_matrix)
            self.loss = self.pos_loss + self.neg_loss_1 + self.neg_loss_2 + self.cocluster_loss
        optimizer = tf.train.AdamOptimizer(config.lr_dis)
        self.d_updates = optimizer.minimize(self.loss)


class HeGAN:
    def __init__(self, args, config, ds, ccl_loss_structures):
        self.args = args
        self.config_hegan = config
        self.ds = ds
        self.ccl_loss_structs = ccl_loss_structures

        # for tracking perf metrics
        self.metrics = {'nmi': list(),
                        'ari': list(),
                        'macrof1': list(),
                        'microf1': list()}
        self.epoch_num = list()
        self.output_embs_dis = None

        print('HeGAN: reading graph...')
        self.n_node, self.n_relation, self.graph = HeGAN_utils.read_graph(config.graph_filename)
        self.node_list = list(self.graph.keys())

        self.node_embed_init_d = HeGAN_utils.read_embeddings(filename=self.config_hegan.pretrain_node_emb_filename_d,
                                                             n_node=self.n_node,
                                                             n_embed=self.config_hegan.n_emb)
        self.node_embed_init_g = HeGAN_utils.read_embeddings(filename=self.config_hegan.pretrain_node_emb_filename_g,
                                                             n_node=self.n_node,
                                                             n_embed=self.config_hegan.n_emb)
        print("... done.")

        print("HeGAN: build model...")
        self.discriminator = None
        self.generator = None
        # ----------------------------------
        # === preparations for the coclustering loss: startconfig
        # ----------------------------------
        if args.cocluster_loss:
            pos_instances = ccl_loss_structures['pos_instances']
            neg_instances = ccl_loss_structures['neg_instances']
            corruption_positions = ccl_loss_structures['corruption_positions']
        else:
            pos_instances = None
            neg_instances = None
            corruption_positions = None
        # ----------------------------------
        # === preparations for the coclustering loss: start
        # ----------------------------------
        self.build_generator()
        self.build_discriminator(pos_instances, neg_instances, corruption_positions)

        self.saver = tf.train.Saver()

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess = tf.Session(config=self.config)
        self.sess.run(self.init_op)

        self.show_config()

    def show_config(self):
        print('--------------------')
        print('Model config : ')
        print('batch_size = ', self.config_hegan.batch_size)
        print('lambda_gen = ', self.config_hegan.lambda_gen)
        print('lambda_dis = ', self.config_hegan.lambda_dis)
        print('n_sample = ', self.config_hegan.n_sample)
        print('lr_gen = ', self.config_hegan.lr_gen)
        print('lr_dis = ', self.config_hegan.lr_dis)
        print('n_epoch = ', self.config_hegan.n_epoch)
        print('d_epoch = ', self.config_hegan.d_epoch)
        print('g_epoch = ', self.config_hegan.g_epoch)
        print('n_emb = ', self.config_hegan.n_emb)
        print('sig = ', self.config_hegan.sig)
        print('hidden_dim = ', self.config_hegan.hidden_dim)
        print('--------------------')

    def build_generator(self):
        self.generator = Generator(n_node=self.n_node,
                                   n_relation=self.n_relation,
                                   node_emd_init=self.node_embed_init_g,
                                   relation_emd_init=None,
                                   hidden_dim=self.config_hegan.hidden_dim,
                                   config=self.config_hegan)

    def build_discriminator(self, pos_instances=None, neg_instances=None, corruption_pos=None):
        self.discriminator = Discriminator(n_node=self.n_node,
                                           n_relation=self.n_relation,
                                           node_emd_init=self.node_embed_init_d,
                                           relation_emd_init=None,
                                           hidden_dim=self.config_hegan.hidden_dim,
                                           pos_instances=pos_instances,
                                           neg_instances=neg_instances,
                                           corruption_pos=corruption_pos,
                                           config=self.config_hegan)

    def train(self):
        print('start training...')
        for epoch in range(self.config_hegan.n_epoch):
            one_epoch_gen_loss = 0
            one_epoch_dis_loss = 0
            print('epoch ' + str(epoch))
            if (epoch + 1) % self.args.downstream_eval_freq == 0 and epoch != 0:
                print('--> evaluating downstream tasks...')
                embedding_matrix = self.sess.run(self.discriminator.node_embedding_matrix)
                self.epoch_num.append(epoch + 1)
                nmi, ari, microf1, macrof1 = evaluate_clu_cla_GTN_NSHE_datasets(dataset=self.ds, embeddings=embedding_matrix,
                                                                                verbose=False)
                self.metrics['nmi'].append(nmi)
                self.metrics['ari'].append(ari)
                self.metrics['microf1'].append(microf1)
                self.metrics['macrof1'].append(macrof1)
                print("this epoch's NMI : " + str(nmi))
                print("this epoch's ARI : " + str(ari))
                print('--> done!')

            for d_epoch in tqdm(range(self.config_hegan.d_epoch)):
                np.random.shuffle(self.node_list)
                one_epoch_dis_loss = 0.0
                one_epoch_pos_loss = 0.0
                one_epoch_neg_loss_1 = 0.0
                one_epoch_neg_loss_2 = 0.0

                for index in range(int(len(self.node_list) / self.config_hegan.batch_size)):
                    pos_node_ids, pos_relation_ids, pos_node_neighbor_ids, neg_node_ids_1, neg_relation_ids_1, neg_node_neighbor_ids_1, neg_node_ids_2, neg_relation_ids_2, node_fake_neighbor_embedding = self.prepare_data_for_d(
                        index)

                    _, dis_loss, pos_loss, neg_loss_1, neg_loss_2 = self.sess.run(
                        [self.discriminator.d_updates, self.discriminator.loss, self.discriminator.pos_loss,
                         self.discriminator.neg_loss_1, self.discriminator.neg_loss_2],
                        feed_dict={self.discriminator.pos_node_id: np.array(pos_node_ids),
                                   self.discriminator.pos_relation_id: np.array(pos_relation_ids),
                                   self.discriminator.pos_node_neighbor_id: np.array(pos_node_neighbor_ids),
                                   self.discriminator.neg_node_id_1: np.array(neg_node_ids_1),
                                   self.discriminator.neg_relation_id_1: np.array(neg_relation_ids_1),
                                   self.discriminator.neg_node_neighbor_id_1: np.array(neg_node_neighbor_ids_1),
                                   self.discriminator.neg_node_id_2: np.array(neg_node_ids_2),
                                   self.discriminator.neg_relation_id_2: np.array(neg_relation_ids_2),
                                   self.discriminator.node_fake_neighbor_embedding: np.array(
                                       node_fake_neighbor_embedding)})
                    one_epoch_dis_loss += dis_loss
                    one_epoch_pos_loss += pos_loss
                    one_epoch_neg_loss_1 += neg_loss_1
                    one_epoch_neg_loss_2 += neg_loss_2

            # G-step

            for g_epoch in tqdm(range(self.config_hegan.g_epoch)):
                np.random.shuffle(self.node_list)
                one_epoch_gen_loss = 0.0

                for index in range(int(len(self.node_list) / self.config_hegan.batch_size)):
                    gen_node_ids, gen_relation_ids, gen_noise_embedding, gen_dis_node_embedding, gen_dis_relation_embedding = self.prepare_data_for_g(
                        index)

                    _, gen_loss = self.sess.run([self.generator.g_updates, self.generator.loss],
                                                feed_dict={self.generator.node_id: np.array(gen_node_ids),
                                                           self.generator.relation_id: np.array(gen_relation_ids),
                                                           self.generator.noise_embedding: np.array(
                                                               gen_noise_embedding),
                                                           self.generator.dis_node_embedding: np.array(
                                                               gen_dis_node_embedding),
                                                           self.generator.dis_relation_embedding: np.array(
                                                               gen_dis_relation_embedding)})

                    one_epoch_gen_loss += gen_loss
            print('Epoch ' + str(epoch) + ' done.')
            print('Discriminator one-epoch loss: ' + str(one_epoch_dis_loss))
            print('Generator one-epoch loss: ' + str(one_epoch_gen_loss))
        self.output_embs_dis = self.sess.run(self.discriminator.node_embedding_matrix)

    def prepare_data_for_d(self, index):

        pos_node_ids = []
        pos_relation_ids = []
        pos_node_neighbor_ids = []

        # real node and wrong relation
        neg_node_ids_1 = []
        neg_relation_ids_1 = []
        neg_node_neighbor_ids_1 = []

        # fake node and true relation
        neg_node_ids_2 = []
        neg_relation_ids_2 = []
        # node_fake_neighbor_embedding = None

        for node_id in self.node_list[index * self.config_hegan.batch_size: (index + 1) * self.config_hegan.batch_size]:
            for i in range(self.config_hegan.n_sample):

                # sample real node and true relation
                relations = list(self.graph[node_id].keys())
                relation_id = relations[np.random.randint(0, len(relations))]
                neighbors = self.graph[node_id][relation_id]
                node_neighbor_id = neighbors[np.random.randint(0, len(neighbors))]

                pos_node_ids.append(node_id)
                pos_relation_ids.append(relation_id)
                pos_node_neighbor_ids.append(node_neighbor_id)

                # sample real node and wrong relation
                neg_node_ids_1.append(node_id)
                neg_node_neighbor_ids_1.append(node_neighbor_id)
                neg_relation_id_1 = np.random.randint(0, self.n_relation)
                while neg_relation_id_1 == relation_id:
                    neg_relation_id_1 = np.random.randint(0, self.n_relation)
                neg_relation_ids_1.append(neg_relation_id_1)

                # sample fake node and true relation
                neg_node_ids_2.append(node_id)
                neg_relation_ids_2.append(relation_id)

        # generate fake node
        noise_dim = self.config_hegan.n_emb if self.config_hegan.hidden_dim == -1 else self.config_hegan.hidden_dim
        noise_embedding = np.random.normal(0.0, self.config_hegan.sig, (len(neg_node_ids_2), noise_dim))

        node_fake_neighbor_embedding = self.sess.run(self.generator.node_neighbor_embedding,
                                                     feed_dict={self.generator.node_id: np.array(neg_node_ids_2),
                                                                self.generator.relation_id: np.array(
                                                                    neg_relation_ids_2),
                                                                self.generator.noise_embedding: np.array(
                                                                    noise_embedding)})

        return pos_node_ids, pos_relation_ids, pos_node_neighbor_ids, neg_node_ids_1, neg_relation_ids_1, \
               neg_node_neighbor_ids_1, neg_node_ids_2, neg_relation_ids_2, node_fake_neighbor_embedding

    def prepare_data_for_g(self, index):
        node_ids = []
        relation_ids = []

        for node_id in self.node_list[index * self.config_hegan.batch_size: (index + 1) * self.config_hegan.batch_size]:
            for i in range(self.config_hegan.n_sample):
                relations = list(self.graph[node_id].keys())
                relation_id = relations[np.random.randint(0, len(relations))]

                node_ids.append(node_id)
                relation_ids.append(relation_id)

        noise_dim = self.config_hegan.n_emb if self.config_hegan.hidden_dim == -1 else self.config_hegan.hidden_dim
        noise_embedding = np.random.normal(0.0, self.config_hegan.sig, (len(node_ids), noise_dim))

        dis_node_embedding, dis_relation_embedding = self.sess.run(
            [self.discriminator.pos_node_embedding, self.discriminator.pos_relation_embedding],
            feed_dict={self.discriminator.pos_node_id: np.array(node_ids),
                       self.discriminator.pos_relation_id: np.array(relation_ids)})
        return node_ids, relation_ids, noise_embedding, dis_node_embedding, dis_relation_embedding
