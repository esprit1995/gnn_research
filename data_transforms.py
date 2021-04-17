import pandas as pd
import numpy as np
import torch
import dgl
import os
import pickle
from datasets import DBLP_MAGNN, IMDB_ACM_DBLP_from_GTN, ACM_HAN, DBLP_ACM_IMDB_from_NSHE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from utils.tools import node_type_encoding


# #############################################################
# Architecture: RGCN. Datasets from papers: MAGNN, GTN, NSHE  #
# Function name format: PAPER_for_architecture
# #############################################################

def MAGNN_for_rgcn():
    """
    transform the datasets.DBLP_MAGNN torch_geometric dataset to a
    torch_geometric.nn.conv.RGCNConv - compatible set of data structures
    :return:
    """
    dataset = DBLP_MAGNN(root="/home/ubuntu/msandal_code/PyG_playground/data/DBLP_MAGNN", use_MAGNN_init_feats=True)[0]

    # reindexing nodes to ensure unique ids;
    # creating id-type mask
    node_types = ['author', 'paper', 'term']
    node_type_id_dict = {node_types[i]: i for i in range(len(node_types))}
    n_nodes_dict = dict()
    id_type_mask = list()
    new_index = 0
    for node_type in node_types:
        dataset.node_id_bag_of_words[node_type]['global_id'] = range(new_index, new_index +
                                                                     dataset.node_id_bag_of_words[node_type].shape[0])
        new_index = new_index + dataset.node_id_bag_of_words[node_type].shape[0]
        n_nodes_dict[node_type] = dataset.node_id_bag_of_words[node_type].shape[0]
        id_type_mask = id_type_mask + [node_type_id_dict[node_type]] * dataset.node_id_bag_of_words[node_type].shape[0]
    id_type_mask = torch.tensor(id_type_mask)

    # creating labels dictionary
    node_labels_dict = dict()
    node_labels_dict['author'] = dataset.id_label['author']['label'].tolist()

    # creating feature matrix. Dimensionality is inferred from term frame (since it is smallest)
    dimensions = dataset.initial_embeddings['term'][1][1].size
    author_feats = np.array([dataset.initial_embeddings['author'][i][1]
                             for i in range(len(dataset.initial_embeddings['author']))])
    paper_feats = np.array([dataset.initial_embeddings['paper'][i][1]
                            for i in range(len(dataset.initial_embeddings['paper']))])
    term_feats = np.array([dataset.initial_embeddings['term'][i][1]
                           for i in range(len(dataset.initial_embeddings['term']))])
    pca = PCA(n_components=dimensions)
    author_feats = pca.fit_transform(author_feats)
    paper_feats = pca.fit_transform(paper_feats)
    node_feature_matrix = torch.tensor(np.vstack((author_feats, paper_feats, term_feats)))
    assert node_feature_matrix.shape[0] == new_index, \
        "#nodes != #features: " + str(node_feature_matrix.shape[0]) + " != " + str(new_index)

    # creating edge index.
    dataset.edge_index_dict[('paper', 'author')] = dataset.edge_index_dict[('paper', 'author')] \
        .merge(dataset.node_id_bag_of_words['author'][['author_id', 'global_id']],
               on='author_id', how='inner') \
        .rename(columns={'global_id': 'author_global_id'}) \
        .merge(dataset.node_id_bag_of_words['paper'][['paper_id', 'global_id']],
               on='paper_id', how='inner') \
        .rename(columns={'global_id': 'paper_global_id'}) \
        .drop(columns=['paper_id', 'author_id'])

    dataset.edge_index_dict[('paper', 'term')] = dataset.edge_index_dict[('paper', 'term')] \
        .merge(dataset.node_id_bag_of_words['term'][['term_id', 'global_id']],
               on='term_id', how='inner') \
        .rename(columns={'global_id': 'term_global_id'}) \
        .merge(dataset.node_id_bag_of_words['paper'][['paper_id', 'global_id']],
               on='paper_id', how='inner') \
        .rename(columns={'global_id': 'paper_global_id'}) \
        .drop(columns=['paper_id', 'term_id'])

    edge_type_id_dict = {('paper', 'author'): 0, ('paper', 'term'): 1}
    source_nodes = np.array(dataset.edge_index_dict[('paper', 'author')]['paper_global_id'].tolist() +
                            dataset.edge_index_dict[('paper', 'term')]['paper_global_id'].tolist())
    target_nodes = np.array(dataset.edge_index_dict[('paper', 'author')]['author_global_id'].tolist() +
                            dataset.edge_index_dict[('paper', 'term')]['term_global_id'].tolist())
    edge_index = torch.tensor([source_nodes, target_nodes])
    edge_type = torch.tensor(np.array([0] * dataset.edge_index_dict[('paper', 'author')].shape[0] +
                                      [1] * dataset.edge_index_dict[('paper', 'term')].shape[0]))
    return n_nodes_dict, node_labels_dict, id_type_mask, node_feature_matrix, edge_index, edge_type


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
                                     redownload=args.redownload_data)[0]

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


def NSHE_for_rgcn(name: str, data_dir: str = '/home/ubuntu/msandal_code/PyG_playground/data/NSHE'):
    """
    prepare datasets from the NSHE paper for the RGCN network
    :param name: name of the dataset. must be one of ['dblp', 'imdb', 'acm']
    :param data_dir: directory where the NSHE datasets are stored/should be downloaded to
    :return: the necessary data structures
    """
    name = name.lower()
    ds = DBLP_ACM_IMDB_from_NSHE(root=data_dir, name=name)[0]

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
# Architecture: GTN. Datasets from papers: GTN                #
# #############################################################

def GTN_for_gtn(name: str, data_dir: str = '/home/ubuntu/msandal_code/PyG_playground/data/IMDB_ACM_DBLP'):
    """
    prepare data structures for GTN architecture
    https://github.com/seongjunyun/Graph_Transformer_Networks
    :param name: name of the dataset. must be one of ['ACM', 'IMDB', 'DBLP']
    :param data_dir: directory where IMDB_ACM_DBLP is stored. If doesn't exist, will be created
    :return:
    """
    if name not in ['ACM', 'IMDB', 'DBLP']:
        raise ValueError('invalid dataset name: ', name)

    dataset = IMDB_ACM_DBLP_from_GTN(root=os.path.join(data_dir, name), name=name)[0]

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

    # id_type_mask, node_feature_matrix
    id_type_mask = dataset['node_type_mask']

    with open(os.path.join(data_dir, name, 'raw', 'node_features.pkl'), 'rb') as f:
        node_features = pickle.load(f)
    with open(os.path.join(data_dir, name, 'raw', 'edges.pkl'), 'rb') as f:
        edges = pickle.load(f)
    with open(os.path.join(data_dir, name, 'raw', 'labels.pkl'), 'rb') as f:
        labels = pickle.load(f)
    num_nodes = edges[0].shape[0]
    num_classes = torch.max(torch.from_numpy(np.array(labels[0])[:, 1]).type(torch.LongTensor)).item() + 1

    node_features = torch.from_numpy(node_features).type(torch.FloatTensor)
    for i, edge in enumerate(edges):
        if i == 0:
            A = torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)
        else:
            A = torch.cat([A, torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
    A = torch.cat([A, torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)

    return A, node_features, num_classes, edge_index, edge_type, id_type_mask


def ACM_HAN_for_han(data_dir: str = '/home/ubuntu/msandal_code/PyG_playground/data/ACM_HAN'):
    dataset = ACM_HAN(root=data_dir)[0]
    metapaths = [('pa', 'ap'), ('pf', 'fp')]
    edge_index_list = list()
    for metapath in metapaths:
        metagraph = dgl.metapath_reachable_graph(dataset.dgl_hetgraph, metapath=metapath)
        edge_index_list.append(metagraph.adjacency_matrix(etype=metagraph.etypes[0]).coalesce().indices())
    id_type_mask = torch.tensor([0] * dataset.features.shape[0])
    return dataset.dgl_hetgraph, dataset.features, dataset.labels, dataset.num_classes, edge_index_list, id_type_mask
