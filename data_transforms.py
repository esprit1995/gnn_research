import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
import os
import pickle

from pathlib import Path
from competitors_perf.competitors_perf_helper_funcs import NSHE_or_GTN_dataset_for_HeGAN
from datasets import IMDB_ACM_DBLP_from_GTN, DBLP_ACM_IMDB_from_NSHE
from utils.tools import node_type_encoding
from utils.tools import normalize_adj, sparse_mx_to_torch_sparse_tensor
from utils.NSHE_utils import gen_neg_edges, gen_ns_instances
from utils.MAGNN_utils import read_data


# #############################################################
# Architecture: RGCN. Datasets from papers: MAGNN, GTN, NSHE  #
# Function name format: PAPER_for_architecture
# #############################################################


def GTN_for_rgcn(name: str, args):
    """
    transform the datasets.IMDB_ACM_DBLP torch_geometric dataset to a
    torch_geometric.nn.conv.RGCNConv - compatible set of data structures
    :param name: name of the dataset to fetch. must be one of ['ACM', 'DBLP', 'IMDB']
    :param args: run arguments
    :return:
    """
    dataset = IMDB_ACM_DBLP_from_GTN(root="/home/ubuntu/msandal_code/PyG_playground/data/IMDB_ACM_DBLP/" + name,
                                     name=name,
                                     multi_type_labels=args.multitype_labels,
                                     redownload=args.redownload_data,
                                     initial_embs=args.acm_dblp_from_gtn_initial_embs)[0]

    # n_nodes_dict
    node_count_info = pd.Series(dataset['node_type_mask']).value_counts()
    n_nodes_dict = {str(val): node_count_info.loc[val] for val in node_count_info.index}

    # id_type_mask, node_feature_matrix
    id_type_mask = dataset['node_type_mask']
    node_feature_matrix = node_type_encoding(dataset['node_features'].numpy(), id_type_mask.numpy())

    # node_labels_dict
    node_labels_dict = {ds_part: dataset[ds_part + '_id_label'] for ds_part in
                        ['train', 'valid', 'test']}

    # edge_index, edge_type
    edge_index = list()
    edge_type = list()
    edge_type_counter = 0
    for key in dataset['edge_index_dict'].keys():
        edge_index.append(dataset['edge_index_dict'][key])
        edge_type = edge_type + [edge_type_counter] * dataset['edge_index_dict'][key].shape[1]
        edge_type_counter += 1
    edge_index = torch.cat(edge_index, dim=1)
    edge_type = torch.tensor(edge_type)

    return dataset, n_nodes_dict, node_labels_dict, id_type_mask, node_feature_matrix, edge_index, edge_type


def NSHE_for_rgcn(name: str, args, data_dir: str = '/home/ubuntu/msandal_code/PyG_playground/data/NSHE'):
    """
    prepare datasets from the NSHE paper for the RGCN network
    :param name: name of the dataset. must be one of ['dblp', 'imdb', 'acm']
    :param args: arguments of the run
    :param data_dir: directory where the NSHE datasets are stored/should be downloaded to
    :return: the necessary data structures
    """
    name = name.lower()
    ds = DBLP_ACM_IMDB_from_NSHE(root=data_dir, name=name, redownload=args.redownload_data)[0]

    # n_nodes_dict
    node_count_info = pd.Series(ds['node_type_mask']).value_counts()
    n_nodes_dict = {str(val): node_count_info.loc[val] for val in node_count_info.index}

    # id_type_mask, node_feature_matrix
    id_type_mask = ds['node_type_mask']
    node_feature_matrix = node_type_encoding(ds['node_features'].numpy(), id_type_mask.numpy())

    # edge_index, edge_type
    edge_index = list()
    edge_type = list()
    edge_type_counter = 0
    for key in ds['edge_index_dict'].keys():
        edge_index.append(ds['edge_index_dict'][key])
        edge_type = edge_type + [edge_type_counter] * ds['edge_index_dict'][key].shape[1]
        edge_type_counter += 1
    edge_index = torch.cat(edge_index, dim=1)
    edge_type = torch.tensor(edge_type)

    # node_labels_dict
    node_labels_dict = {ds_part: ds[ds_part + '_id_label'] for ds_part in
                        ['train', 'valid', 'test']}

    return ds, n_nodes_dict, node_labels_dict, id_type_mask, node_feature_matrix, edge_index, edge_type


# #############################################################
# Architecture: GTN. Datasets from papers: GTN, NSHE          #
# Naming convention format: PAPER_for_architecture
# #############################################################

def NSHE_for_gtn(args, data_dir: str = '/home/ubuntu/msandal_code/PyG_playground/data/NSHE/'):
    """
    prepare data structures for GTN architecture: NSHE article datasets
    :param args: experiment run arguments
    :param data_dir: directory where NSHE datasets are/will be stored
    :return:
    """
    name = args.dataset
    if name.upper() not in ['ACM', 'IMDB', 'DBLP']:
        raise ValueError('invalid dataset name: ', name)

    dataset = DBLP_ACM_IMDB_from_NSHE(root=data_dir, name=name.lower())[0]

    # edge_index, edge_type
    edge_index = list()
    edge_type = list()
    edge_type_counter = 0
    for key in list(dataset['edge_index_dict'].keys()):
        edge_index.append(dataset['edge_index_dict'][key])
        edge_type = edge_type + [edge_type_counter] * dataset['edge_index_dict'][key].shape[1]
        edge_type_counter += 1
    edge_index = torch.cat(edge_index, dim=1)
    edge_type = torch.tensor(edge_type)

    # node_labels_dict
    node_labels_dict = {ds_part: dataset[ds_part + '_id_label'] for ds_part in
                        ['train', 'valid', 'test']}

    # id_type_mask, node_features
    id_type_mask = dataset['node_type_mask']
    node_features = dataset['node_features']
    num_classes = np.unique(dataset['train_id_label'][:, 1]).size

    # adjacency tensor
    n_nodes = id_type_mask.shape[0]

    def edge_index_to_dense_adj_matrix(n_nodes: int, edge_index: np.array):
        adj_matrix = np.zeros((n_nodes, n_nodes))
        adj_matrix[edge_index[0], edge_index[1]] = 1
        return adj_matrix

    A = None
    for edge_type in list(dataset['edge_index_dict'].keys()):
        if A is None:
            A = torch.from_numpy(
                edge_index_to_dense_adj_matrix(n_nodes, dataset['edge_index_dict'][edge_type].numpy())).unsqueeze(-1)
        else:
            A = torch.cat([A, torch.from_numpy(
                edge_index_to_dense_adj_matrix(n_nodes, dataset['edge_index_dict'][edge_type].numpy())).unsqueeze(-1)],
                          dim=-1)
    A = torch.cat([A, torch.eye(n_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
    return A, node_labels_dict, node_features, num_classes, edge_index, edge_type, id_type_mask, dataset


def GTN_for_gtn(args, data_dir: str = '/home/ubuntu/msandal_code/PyG_playground/data/IMDB_ACM_DBLP'):
    """
    prepare data structures for GTN architecture: GTN article datasets
    https://github.com/seongjunyun/Graph_Transformer_Networks
    :param args: experiment run arguments
    :param data_dir: directory where IMDB_ACM_DBLP is stored. If doesn't exist, will be created
    :return:
    """
    name = args.dataset
    if name not in ['ACM', 'IMDB', 'DBLP']:
        raise ValueError('invalid dataset name: ', name)

    dataset = IMDB_ACM_DBLP_from_GTN(root=os.path.join(data_dir, name),
                                     name=name,
                                     redownload=args.redownload_data,
                                     initial_embs=args.acm_dblp_from_gtn_initial_embs)[0]

    # edge_index, edge_type
    edge_index = list()
    edge_type = list()
    edge_type_counter = 0
    for key in list(dataset['edge_index_dict'].keys()):
        edge_index.append(dataset['edge_index_dict'][key])
        edge_type = edge_type + [edge_type_counter] * dataset['edge_index_dict'][key].shape[1]
        edge_type_counter += 1
    edge_index = torch.cat(edge_index, dim=1)
    edge_type = torch.tensor(edge_type)

    # node_labels_dict
    node_labels_dict = {ds_part: dataset[ds_part + '_id_label'] for ds_part in
                        ['train', 'valid', 'test']}

    # id_type_mask, node_feature_matrix
    id_type_mask = dataset['node_type_mask']

    node_features = dataset['node_features']
    with open(os.path.join(data_dir, name, 'raw', 'edges.pkl'), 'rb') as f:
        edges = pickle.load(f)
    with open(os.path.join(data_dir, name, 'raw', 'labels.pkl'), 'rb') as f:
        labels = pickle.load(f)
    num_nodes = edges[0].shape[0]
    num_classes = torch.max(torch.from_numpy(np.array(labels[0])[:, 1]).type(torch.LongTensor)).item() + 1

    for i, edge in enumerate(edges):
        if i == 0:
            A = torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)
        else:
            A = torch.cat([A, torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
    A = torch.cat([A, torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)

    return A, node_labels_dict, node_features, num_classes, edge_index, edge_type, id_type_mask, dataset


# #############################################################
# Architecture: NSHE. Datasets from papers: GTN, NSHE         #
# Naming convention format: PAPER_for_architecture
# #############################################################
E_NEG_RATE = 1  # was always 1 in the original NSHE code
NS_NEG_RATE = 4  # was always 4 in the original NSHE code


class GTN_or_NSHE_for_nshe(object):
    """
    prepare data structures for NSHE architecture: GTN article datasets
    https://github.com/seongjunyun/Graph_Transformer_Networks
    Has to be a class and not a function due to the necessity to keep an internal state
    """

    def __init__(self,
                 args,
                 data_dir: str = '/home/ubuntu/msandal_code/PyG_playground/data/IMDB_ACM_DBLP',
                 nshe_personal_storage: str = '/home/ubuntu/msandal_code/PyG_playground/data/model_data/NSHE'):
        """
        :param args: experiment run arguments
        :param data_dir: directory where IMDB_ACM_DBLP is stored. If doesn't exist, will be created
        :param nshe_personal_storage: NSHE saves sampled schema instances for every epoch. this variable
                               indicates the path where the files will be saved to
        """
        name = args.dataset
        if name not in ['ACM', 'DBLP']:
            raise ValueError('invalid dataset name: ', name)
        # ----> remark: NSHE is 'traditionally' run on initial DeepWalk embeddings
        if args.from_paper == 'GTN':
            dataset = IMDB_ACM_DBLP_from_GTN(root=os.path.join(data_dir, name),
                                             name=name,
                                             redownload=args.redownload_data,
                                             initial_embs='deepwalk')[0]
        elif args.from_paper == 'NSHE':
            dataset = DBLP_ACM_IMDB_from_NSHE(root=data_dir,
                                              name=name.lower())[0]
        else:
            raise ValueError('Cannot prepare datasets from ' + str(args.from_paper) + ': unimplemented')
        self.ds = dataset
        self.seed = args.random_seed
        self.dataset_name = args.dataset
        self.nshe_personal_storage = nshe_personal_storage
        # --> initialize the required fields as in the original code
        self.node_id, self.t_info, self.node_cnt_all, self.edge_cnt = None, None, None, None
        self.adj, self.adj2, self.dw_features, self.edge = None, None, None, None
        self.true_feature, self.feature = {}, {}
        self.ns_instances, self.ns_label = None, None
        self.node_types = None
        self.optimizer = None
        self.ns_neg_rate = 4
        self.seed_set = []

        # --> edge index and type mask for triplet loss
        self.edge_index = list()
        self.edge_type = list()
        edge_type_counter = 0
        for key in list(self.ds['edge_index_dict'].keys()):
            self.edge_index.append(dataset['edge_index_dict'][key])
            self.edge_type = self.edge_type + [edge_type_counter] * self.ds['edge_index_dict'][key].shape[1]
            edge_type_counter += 1
        self.edge_index = torch.cat(self.edge_index, dim=1)
        self.edge_type = torch.tensor(self.edge_type)

        self.id_type_mask = dataset['node_type_mask']

        # --> node label dict for evaluation
        self.node_label_dict = {ds_part: self.ds[ds_part + '_id_label'] for ds_part in
                                ['train', 'valid', 'test']}

        # --> load the network: use data structures required by the original NSHE
        node_type_mask = dataset['node_type_mask'].numpy()
        int_to_node_type_dict = {'ACM': {'0': 'p', '1': 'a', '2': 's'},
                                 'DBLP': {'0': 'a', '1': 'p', '2': 'c'}}
        self.node_id = dict()
        self.t_info = dict()
        self.node2id = dict()
        self.id2node = dict()

        for node_int_type in list(int_to_node_type_dict[self.dataset_name].keys()):
            node_letter_type = int_to_node_type_dict[self.dataset_name][node_int_type]
            corresp_node_ids = np.argwhere(node_type_mask == int(node_int_type)).reshape(-1)
            self.node_id[node_letter_type] = \
                {node_letter_type + str(i): corresp_node_ids[i] for i in range(corresp_node_ids.size)}
            self.node2id = {**self.node2id, **self.node_id[node_letter_type]}
            self.id2node = {**self.id2node,
                            **{corresp_node_ids[i]: node_letter_type + str(i) for i in range(corresp_node_ids.size)}}
            self.t_info[node_letter_type] = {'cnt': corresp_node_ids.size,
                                             'ind': range(np.min(corresp_node_ids), np.max(corresp_node_ids) + 1)}
        self.node_types = self.node_id.keys()

        # --> load the edges and adjacency info
        # self.adj, self.edge, self.edge_cnt, self.adj2
        node_cnt = len(self.node2id)
        edge_index_dict = dataset['edge_index_dict']
        rows = list()
        cols = list()

        # necessary because original code does not take symmetric edges into account
        unique_edge_types = list(edge_index_dict.keys())
        for edge_type in list(edge_index_dict.keys()):
            if edge_type in unique_edge_types and (edge_type[1], edge_type[0]) in unique_edge_types:
                unique_edge_types.remove(edge_type)
        for edge_type in unique_edge_types:
            rows = rows + edge_index_dict[edge_type][0].tolist()
            cols = cols + edge_index_dict[edge_type][1].tolist()
        adj = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(node_cnt, node_cnt))
        self.adj2 = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        del adj
        _ = rows.copy()
        rows = rows + cols
        cols = cols + _
        self.edge = {"r": rows, "c": cols}
        adj_normalized = normalize_adj(self.adj2 + sp.eye(self.adj2.shape[0]))
        self.adj = sparse_mx_to_torch_sparse_tensor(adj_normalized)
        self.edge_cnt = len(rows)

        # --> set the features
        self.dw_features = dataset['node_features'].numpy()
        self.true_feature = {t: None for t in self.node_types}
        for t in self.node_types:
            self.feature[t] = torch.FloatTensor(self.dw_features[self.t_info[t]['ind']])

        # --> sampling

        for i in range(args.epochs):
            self.seed_set.append(np.random.randint(1000))

    def get_epoch_samples(self, epoch, args):
        """
        Renew ns_instances and neg_edges in every epoch:
        1. get the seed for current epoch
        2. find using seed
            Y: load the file
            N: sample again and save
        """

        # seed for current epoch
        epoch_seed = self.seed_set[epoch]
        np.random.seed(epoch_seed)

        def _get_neg_edge(epoch_seed_):
            fname = os.path.join(self.nshe_personal_storage, 'reusable', self.dataset_name + '_' + args.from_paper,
                                 "neg_edges",
                                 "NE-rate=" + str(E_NEG_RATE) + "_seed=" + str(epoch_seed_) + '.dat')
            if os.path.exists(fname):
                # load
                with open(fname, 'rb') as handle:
                    try:
                        epoch_data = pickle.load(handle)
                        self.neg_edge = epoch_data['neg_edge']
                    except EOFError:
                        os.remove(fname)
                        print(epoch_seed_, fname)
            else:
                # sample
                self.neg_edge = gen_neg_edges(self.adj2, self.edge, E_NEG_RATE)
                # save
                data_to_save = {'neg_edge': self.neg_edge}
                Path('/'.join(str(fname).split('/')[:-1])).mkdir(parents=True, exist_ok=True)
                with open(fname, 'wb') as handle:
                    pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

        def _get_ns_instance(epoch_seed_):
            fname = os.path.join(self.nshe_personal_storage,
                                 'reusable', self.dataset_name + '_' + args.from_paper,
                                 "network_schema_instances",
                                 "NS-rate=" + str(NS_NEG_RATE) + "_seed=" + str(epoch_seed_) + '.dat')
            if os.path.exists(fname):
                # load
                with open(fname, 'rb') as handle:
                    try:
                        epoch_data = pickle.load(handle)
                    except EOFError:
                        print(epoch_seed_, fname)
                self.ns_instances = epoch_data['ns_instances']
                self.ns_label = epoch_data['ns_label']
            else:

                f_type_adj = os.path.join(self.nshe_personal_storage,
                                          self.dataset_name + '_' + args.from_paper,
                                          'relation2id.txt')
                self.ns_instances, self.ns_label = gen_ns_instances(f_type_adj, self.adj2, self.edge, self.t_info,
                                                                    self.ns_neg_rate)
                # save
                data_to_save = {
                    'ns_instances': self.ns_instances,
                    'ns_label': self.ns_label}
                Path('/'.join(str(fname).split('/')[:-1])).mkdir(parents=True, exist_ok=True)
                with open(fname, 'wb') as handle:
                    pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

        _get_neg_edge(epoch_seed)
        _get_ns_instance(epoch_seed)
        return


# #############################################################
# Architecture: MAGNN. Datasets from papers: GTN, NSHE        #
# Naming convention format: PAPER_for_architecture
# #############################################################

def GTN_NSHE_for_MAGNN(args, data_dir: str = '/home/ubuntu/msandal_code/PyG_playground/data/IMDB_ACM_DBLP',
                       magnn_personal_storage: str = '/home/ubuntu/msandal_code/PyG_playground/data/model_data/MAGNN'):
    """
        prepare data structures for MAGNN architecture: GTN article datasets
        https://github.com/seongjunyun/Graph_Transformer_Networks
        :param args: experiment run arguments
        :param data_dir: directory where IMDB_ACM_DBLP is stored. If doesn't exist, will be created
        :param magnn_personal_storage: where to save data needed for MAGNN run. If does not exist, will be created
        :return:
        """
    name = args.dataset
    if name not in ['ACM', 'IMDB', 'DBLP']:
        raise ValueError('invalid dataset name: ', name)

    if args.from_paper == 'GTN':
        dataset = IMDB_ACM_DBLP_from_GTN(root=os.path.join(data_dir, name),
                                         name=str(name).upper(),
                                         redownload=args.redownload_data,
                                         initial_embs=args.acm_dblp_from_gtn_initial_embs)[0]
    elif args.from_paper == 'NSHE':
        dataset = DBLP_ACM_IMDB_from_NSHE(root=data_dir,
                                          name=str(name).lower())[0]
    else:
        raise ValueError('MAGNN cannot be applied to datasets from paper : ' + str(args.from_paper))

    path_to_files = PyG_to_MAGNN_files(dataset,
                                       '_'.join([str(name).lower(), str(args.from_paper).lower()]),
                                       magnn_personal_storage)
    graph_statistics, type_mask, node_labels, node_order, ntype_features, posi_edges, node_mptype_mpinstances = \
        read_data(nodefile=os.path.join(path_to_files, 'node.dat'),
                  linkfile=os.path.join(path_to_files, 'link.dat'),
                  pathfile=os.path.join(path_to_files, 'path.dat'),
                  labelfile=None,  # only needed in supervised learning case, which we never follow
                  attributed='True',  # in our case, the learning is always attributed
                  supervised='False')  # in our case, the learning is always unsupervised
    # yes. the authors of MAGNN in HNE actually used strings instead of booleans. Lol.

    # edge_index, edge_type
    edge_index = list()
    for key in list(dataset['edge_index_dict'].keys()):
        edge_index.append(dataset['edge_index_dict'][key])
    edge_index = torch.cat(edge_index, dim=1)

    # id_type_mask, node_feature_matrix
    id_type_mask = dataset['node_type_mask']

    # node_labels_dict
    node_labels_dict = {ds_part: dataset[ds_part + '_id_label'] for ds_part in
                        ['train', 'valid', 'test']}

    return graph_statistics, type_mask, node_labels, \
           node_order, ntype_features, posi_edges, node_mptype_mpinstances, \
           node_labels_dict, id_type_mask, edge_index, dataset


def PyG_to_MAGNN_files(ds, ds_name, save_to) -> str:
    """
    create files needed for MAGNN run on a given dataset: link.dat and node.dat
    if for the given dataset these files already exist, they will be overwritten
    :param ds: dataset to convert
    :param ds_name: name of the dataset. example: dblp_gtn
    :param save_to: where to save the results to
    :return: directory containing the results
    """
    if not os.path.exists(os.path.join(save_to, ds_name)):
        Path(os.path.join(save_to, ds_name)).mkdir(exist_ok=True, parents=True)
    if os.path.exists(os.path.join(save_to, ds_name, 'link.dat')):
        os.remove(os.path.join(save_to, ds_name, 'link.dat'))
    if os.path.exists(os.path.join(save_to, ds_name, 'node.dat')):
        os.remove(os.path.join(save_to, ds_name, 'node.dat'))
    open(os.path.join(save_to, ds_name, 'link.dat'), 'w').close()
    open(os.path.join(save_to, ds_name, 'node.dat'), 'w').close()

    # create node data
    with open(os.path.join(save_to, ds_name, 'node.dat'), 'a') as f:
        for idx in range(ds['node_features'].shape[0]):
            line = ''
            line += (str(idx) + '\t' + str(ds['node_type_mask'][idx].item()) + '\t')
            for j in range(ds['node_features'].shape[1]):
                line += str(ds['node_features'][idx][j].item()) + ','
            line = line[:-1]
            line = line + '\n'
            f.write(line)

    # create link data
    with open(os.path.join(save_to, ds_name, 'link.dat'), 'a') as f:
        for etype, key in enumerate(list(ds['edge_index_dict'].keys())):
            edge_index = ds['edge_index_dict'][key].numpy()
            for i in range(edge_index.shape[1]):
                line = str(edge_index[0, i]) + '\t' + str(edge_index[1, i]) + '\t' + str(etype)
                line = line + '\n'
                f.write(line)

    # if path file doesn't exist, create it
    # default paths are simply edge types that are present in edge_index_dict of the ds
    if not os.path.exists(os.path.join(save_to, ds_name, 'path.dat')):
        with open(os.path.join(save_to, ds_name, 'path.dat'), 'w+') as f:
            for key in list(ds['edge_index_dict'].keys()):
                f.write(str(key[0]) + '\t' + str(key[1]) + '\n')
    return str(os.path.join(save_to, ds_name))


# #############################################################
# Architecture: HeGAN. Datasets from papers: GTN, NSHE        #
# Naming convention format: PAPER_for_architecture
# #############################################################
def GTN_NSHE_for_HeGAN(args, data_dir: str = '/home/ubuntu/msandal_code/PyG_playground/data/IMDB_ACM_DBLP',
                       hegan_personal_storage: str = '/home/ubuntu/msandal_code/PyG_playground/data/model_data/HeGAN',
                       model_params: dict = None):
    """
    prepare data structures for HeGAN architecture: GTN article datasets
    https://github.com/seongjunyun/Graph_Transformer_Networks
    :param args: experiment run arguments
    :param data_dir: directory where IMDB_ACM_DBLP is stored. If doesn't exist, will be created
    :param hegan_personal_storage: where to save data needed for MAGNN run. If does not exist, will be created
    :param model_params: dictionary containing the model parameters
    :return:
    """
    if model_params is None:
        print('model_params argument should not be None. please provide a valid dictionary')
        return

    class CONFIG_CLASS:
        def __init__(self):
            pass

    config = CONFIG_CLASS()
    name = args.dataset
    root = None
    if name not in ['ACM', 'IMDB', 'DBLP']:
        raise ValueError('invalid dataset name: ', name)

    if args.from_paper == 'GTN':
        root = os.path.join(data_dir, name)
        dataset = IMDB_ACM_DBLP_from_GTN(root=root,
                                         name=str(name).upper(),
                                         redownload=args.redownload_data,
                                         initial_embs=args.acm_dblp_from_gtn_initial_embs)[0]
    elif args.from_paper == 'NSHE':
        root = data_dir
        dataset = DBLP_ACM_IMDB_from_NSHE(root=root,
                                          name=str(name).lower())[0]
    else:
        raise ValueError('HeGAN cannot be applied to datasets from paper : ' + str(args.from_paper))

    # record the model parameters in config
    for key in list(model_params.keys()):
        setattr(config, key, model_params[key])

    # create files necessary for the run of original HeGAN code
    # create corresponding entries in the config object
    path_to_files, graph_filename, pretrain_filename = PyG_to_HeGAN_files(str(name).lower(),
                                                                          str(args.from_paper).lower(),
                                                                          args.acm_dblp_from_gtn_initial_embs,
                                                                          root,
                                                                          hegan_personal_storage,
                                                                          dataset)
    setattr(config, 'graph_filename', os.path.join(path_to_files, graph_filename))
    setattr(config, 'pretrain_node_emb_filename_d', os.path.join(path_to_files, pretrain_filename))
    setattr(config, 'pretrain_node_emb_filename_g', os.path.join(path_to_files, pretrain_filename))

    # record the initial embeddings dim in the config object
    setattr(config, 'n_emb', dataset['node_features'].shape[1])

    # edge_index, edge_type
    edge_index = list()
    for key in list(dataset['edge_index_dict'].keys()):
        edge_index.append(dataset['edge_index_dict'][key])
    edge_index = torch.cat(edge_index, dim=1)

    # id_type_mask, node_feature_matrix
    id_type_mask = dataset['node_type_mask']

    # node_labels_dict
    node_labels_dict = {ds_part: dataset[ds_part + '_id_label'] for ds_part in
                        ['train', 'valid', 'test']}
    return config, node_labels_dict, id_type_mask, edge_index, dataset


def PyG_to_HeGAN_files(ds_name, ds_paper, init_embs, pyg_root, save_to, dataset=None):
    """

    :param ds:
    :param ds_name:
    :param ds_paper:
    :param init_embs:
    :param pyg_root:
    :param save_to:
    :param dataset:
    :return:
    """
    path_to_files = os.path.join(save_to, '_'.join([ds_name.lower(), ds_paper.lower()]))
    if not os.path.exists(path_to_files):
        Path(path_to_files).mkdir(exist_ok=True, parents=True)
    NSHE_or_GTN_dataset_for_HeGAN(name=ds_name.lower(),
                                  from_paper=ds_paper.lower(),
                                  root=pyg_root,
                                  output_dir=path_to_files,
                                  initial_embs=init_embs,
                                  ds_=dataset)
    graph_filename = ds_name.lower() + '_' + ds_paper.lower() + '_triple.dat'
    pretrain_filename = ds_name.lower() + '_' + ds_paper.lower() + '_pre_train.emb'
    return path_to_files, graph_filename, pretrain_filename
