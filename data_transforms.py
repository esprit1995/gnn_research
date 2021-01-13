import pandas as pd
import numpy as np
import torch
from datasets import DBLP_MAGNN
from sklearn.decomposition import PCA


def dblp_for_rgcn():
    """
    transform the datasets.DBLP_MAGNN torch_geometric dataset to a
    torch_geometric.nn.conv.RGCNConv - compatible set of data structures
    :return:
    """
    dataset = DBLP_MAGNN(root="/home/ubuntu/msandal_code/PyG_playground/dblp", use_MAGNN_init_feats=True)[0]

    # reindexing nodes to ensure unique ids;
    # creating id-type mask
    node_types = ['author', 'paper', 'term']
    node_type_id_dict = {node_types[i]: i for i in range(len(node_types))}
    id_type_mask = list()
    new_index = 0
    for node_type in node_types:
        dataset.node_id_bag_of_words[node_type]['global_id'] = range(new_index, new_index +
                                                                     dataset.node_id_bag_of_words[node_type].shape[0])
        new_index = new_index + dataset.node_id_bag_of_words[node_type].shape[0]
        id_type_mask = id_type_mask + [node_type_id_dict[node_type]] * dataset.node_id_bag_of_words[node_type].shape[0]
    id_type_mask = torch.tensor(id_type_mask)
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
    return node_type_id_dict, edge_type_id_dict, id_type_mask, node_feature_matrix, edge_index, edge_type
