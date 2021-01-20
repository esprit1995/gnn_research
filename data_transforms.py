import pandas as pd
import numpy as np
import torch
import os
import pickle
from datasets import DBLP_MAGNN, IMDB_ACM_DBLP
from sklearn.decomposition import PCA
from utils.tools import node_type_encoding


def DBLP_MAGNN_for_rgcn():
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


def IMDB_ACM_DBLP_for_rgcn(name: str):
    """
    transform the datasets.IMDB_ACM_DBLP torch_geometric dataset to a
    torch_geometric.nn.conv.RGCNConv - compatible set of data structures
    :param name: name of the dataset to fetch. must be one of ['ACM', 'DBLP', 'IMDB']
    :return:
    """
    dataset = IMDB_ACM_DBLP(root="/home/ubuntu/msandal_code/PyG_playground/data/IMDB_ACM_DBLP/" + name, name=name)[0]

    # n_nodes_dict
    node_count_info = pd.Series(dataset['node_type_mask']).value_counts()
    n_nodes_dict = {str(val): node_count_info.loc[val] for val in node_count_info.index}

    # id_type_mask, node_feature_matrix
    id_type_mask = dataset['node_type_mask']
    node_feature_matrix = node_type_encoding(dataset['node_features'].numpy(), id_type_mask.numpy())

    # node_labels_dict
    labeled_type = id_type_mask[dataset['train_id_label'][0][0].item()].item()
    node_labels_dict = {str(labeled_type) + '_' + ds_part: dataset[ds_part + '_id_label'] for ds_part in
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

    return n_nodes_dict, node_labels_dict, id_type_mask, node_feature_matrix, edge_index, edge_type


def IMDB_ACM_DBLP_for_gtn(name: str, data_dir: str = '/home/ubuntu/msandal_code/PyG_playground/data/IMDB_ACM_DBLP'):
    """
    prepare data structures for GTN architecture
    https://github.com/seongjunyun/Graph_Transformer_Networks
    :param name: name of the dataset. must be one of ['ACM', 'IMDB', 'DBLP']
    :param data_dir: directory where IMDB_ACM_DBLP is stored. If doesn't exist, will be created
    :return:
    """
    if name not in ['ACM', 'IMDB', 'DBLP']:
        raise ValueError('invalid dataset name: ', name)

    dataset = IMDB_ACM_DBLP(root=os.path.join(data_dir,name), name=name)[0]

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
