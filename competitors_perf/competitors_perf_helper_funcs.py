import pandas as pd
import numpy as np
import os

from datasets import DBLP_ACM_IMDB_from_NSHE, IMDB_ACM_DBLP_from_GTN
from pathlib import Path
from sklearn.decomposition import PCA


# #######################################
# funcs to transform data for specific architecture
# #######################################

def NSHE_or_GTN_dataset_to_edgelist(root='/home/ubuntu/msandal_code/PyG_playground/data/NSHE',
                                    name='dblp',
                                    from_paper='nshe',
                                    output_dir='/home/ubuntu/msandal_code/PyG_playground/competitors_perf/input_for_competitors'):
    """
    helper for the DeepWalk baseline: NSHE graphs  need to be transformed into
    edgelist files
    :param root: where to find/download NSHE dataset in question
    :param name: which dataset to fetch. must be one of ['dblp', 'imdb', 'acm']
    :param from_paper: from which paper to take the datasets. must be either 'nshe' or 'gtn'
    :param output_dir: where to save the resulting edge list
    :return:
    """
    assert str(from_paper).lower() in ['nshe', 'gtn'], \
        'NSHE_dataset_to_edgelist(): invalid paper'
    assert str(name).lower() in ['dblp', 'acm', 'imdb'], \
        'NSHE_dataset_to_edgelist(): invalid dataset name required'
    if from_paper.lower() == 'nshe':
        ds = DBLP_ACM_IMDB_from_NSHE(root=root, name=str(name).lower())[0]
    else:
        ds = IMDB_ACM_DBLP_from_GTN(root=root, name=str(name).upper())[0]
    edge_types = list(ds['edge_index_dict'].keys())
    edge_list = ds['edge_index_dict'][edge_types[0]].numpy().T
    for i in range(1, len(edge_types)):
        edge_list = np.vstack([edge_list, ds['edge_index_dict'][edge_types[i]].numpy().T])
    edge_df = pd.DataFrame(data=edge_list, columns=['id1', 'id2'])
    save_under_name = os.path.join(output_dir, str(name) + '_from_' + str(from_paper).lower() + '_edgelist.txt')
    edge_df.to_csv(save_under_name, sep=' ', header=False, index=False)
    return


def GTN_datasets_for_NSHE_prepa(root='/home/ubuntu/msandal_code/PyG_playground/data/IMDB_ACM_DBLP/',
                                name='dblp',
                                output_dir='/home/ubuntu/msandal_code/PyG_playground/competitors_perf/input_for_competitors'):
    """
    prepare input for the original NSHE network by transforming the PyG datasets into the required set of files
    :param root: path where IMDB_ACM_DBLP_from_GTN data is stored/will be downloaded
    :param name: name of the dataset to transform. must be one of ['acm', 'dblp', 'imdb']
    :param output_dir: where to save the results to
    :return:
    """
    root = os.path.join(root, str(name).upper())
    ds = IMDB_ACM_DBLP_from_GTN(root=root, name=str(name).upper())[0]

    if name == 'dblp':
        # === prepare data structures: node2id, relation2id, relations.
        relation2id = pd.DataFrame({'relation': ['pa', 'pc'],
                                    'code': [0, 1]})
        pa_rels = ds['edge_index_dict'][('1', '0')].numpy().T
        pc_rels = ds['edge_index_dict'][('1', '2')].numpy().T
        relations = pd.DataFrame(data=np.vstack([pa_rels, pc_rels]),
                                 columns=['id1', 'id2'])
        edge_type = [0] * pa_rels.shape[0] + [1] * pc_rels.shape[0]
        relations['edge_type'] = pd.Series(edge_type)
        relations['weird_crap'] = 1
        node2id = pd.DataFrame({'node_code': ['a' + str(i) for i in range(4057)] +
                                             ['p' + str(i) for i in range(14328)] +
                                             ['c' + str(i) for i in range(20)],
                                'node_id': list(range(4057 + 14328 + 20))})
        # === saving
        Path(os.path.join(output_dir, 'dblp_gtn')).mkdir(parents=True, exist_ok=True)
        relations.to_csv(os.path.join(output_dir, 'dblp_gtn', 'relations.txt'),
                         sep='\t',
                         header=False,
                         index=False)
        node2id.to_csv(os.path.join(output_dir, 'dblp_gtn', 'node2id.txt'),
                       sep='\t',
                       header=False,
                       index=False)
        line_prepender(os.path.join(output_dir, 'dblp_gtn', 'node2id.txt'), str(node2id.shape[0]))
        relation2id.to_csv(os.path.join(output_dir, 'dblp_gtn', 'relation2id.txt'),
                           sep='\t',
                           header=False,
                           index=False)
        line_prepender(os.path.join(output_dir, 'dblp_gtn', 'relation2id.txt'), str(relation2id.shape[0]))
        return relation2id, relations, node2id
    elif name == 'acm':
        # === prepare data structures: node2id, relation2id, relations.
        relation2id = pd.DataFrame({'relation': ['pa', 'ps'],
                                    'code': [0, 1]})
        pa_rels = ds['edge_index_dict'][('0', '1')].numpy().T
        ps_rels = ds['edge_index_dict'][('0', '2')].numpy().T
        relations = pd.DataFrame(data=np.vstack([pa_rels, ps_rels]),
                                 columns=['id1', 'id2'])
        edge_type = [0] * pa_rels.shape[0] + [1] * ps_rels.shape[0]
        relations['edge_type'] = pd.Series(edge_type)
        relations['weird_crap'] = 1
        node2id = pd.DataFrame({'node_code': ['p' + str(i) for i in range(3025)] +
                                             ['a' + str(i) for i in range(5912)] +
                                             ['s' + str(i) for i in range(57)],
                                'node_id': list(range(3025 + 5912 + 57))})
        # === saving
        Path(os.path.join(output_dir, 'acm_gtn')).mkdir(parents=True, exist_ok=True)
        relations.to_csv(os.path.join(output_dir, 'acm_gtn', 'relations.txt'),
                         sep='\t',
                         header=False,
                         index=False)
        node2id.to_csv(os.path.join(output_dir, 'acm_gtn', 'node2id.txt'),
                       sep='\t',
                       header=False,
                       index=False)
        line_prepender(os.path.join(output_dir, 'acm_gtn', 'node2id.txt'), str(node2id.shape[0]))
        relation2id.to_csv(os.path.join(output_dir, 'acm_gtn', 'relation2id.txt'),
                           sep='\t',
                           header=False,
                           index=False)
        line_prepender(os.path.join(output_dir, 'acm_gtn', 'relation2id.txt'), str(relation2id.shape[0]))
        return relation2id, relations, node2id
    else:
        raise NotImplementedError('GTN_datasets_for_NSHE_prepa(): invalid dataset requested')


def NSHE_or_GTN_dataset_for_HeGAN(root='/home/ubuntu/msandal_code/PyG_playground/data/NSHE',
                                  name='dblp',
                                  from_paper='nshe',
                                  output_dir='/home/ubuntu/msandal_code/PyG_playground/competitors_perf/input_for_competitors',
                                  initial_embs='original',
                                  ds_=None):
    """
    helper function to prepare datasets from NSHE or GTN papers for embedding by HeGAN
    :param root: where to find/download NSHE dataset in question
    :param name: which dataset to fetch. must be one of ['dblp', 'imdb', 'acm']
    :param from_paper: from which paper to take the datasets. must be either 'nshe' or 'gtn'
    :param output_dir: where to save the resulting edge list
    :param initial_embs: which initial node embeddings to use for datasets from the GTN-paper
    :param ds_: it is also possible to just pass a dataset directly.
    :return:
    """
    if ds_ is None:
        assert str(from_paper).lower() in ['nshe', 'gtn'], \
            'NSHE_or_GTN_dataset_for_HeGAN(): invalid paper'
        assert str(name).lower() in ['dblp', 'acm', 'imdb'], \
            'NSHE_or_GTN_dataset_for_HeGAN(): invalid dataset name required'
        if from_paper.lower() == 'nshe':
            ds = DBLP_ACM_IMDB_from_NSHE(root=root, name=str(name).lower())[0]
        else:
            ds = IMDB_ACM_DBLP_from_GTN(root=root, name=str(name).upper(), initial_embs=initial_embs)[0]
    else:
        ds = ds_
    edge_types = list(ds['edge_index_dict'].keys())
    edge_types_to_int = {edge_types[i]: i for i in range(len(edge_types))}

    # ===> obtain a frame containing graph info in triples: source_id // target_id // edge_type (int)
    triples = pd.DataFrame(data=ds['edge_index_dict'][edge_types[0]].numpy().T, columns=['source_id', 'target_id'])
    triples['edge_type'] = edge_types_to_int[edge_types[0]]
    for i in range(1, len(edge_types)):
        to_append = pd.DataFrame(data=ds['edge_index_dict'][edge_types[i]].numpy().T,
                                 columns=['source_id', 'target_id'])
        to_append['edge_type'] = edge_types_to_int[edge_types[i]]
        triples = triples.append(to_append, ignore_index=True)

    # ===> obtain node features in format: node_id feat_1 feat_2 ... feat_n
    node_features = ds['node_features'].numpy()
    ids = np.array(list(range(node_features.shape[0])))
    node_feature_df = pd.DataFrame(data=np.c_[ids, node_features])

    # ===> save results in folder
    saving_path = output_dir
    Path(saving_path).mkdir(parents=True, exist_ok=True)
    triples.to_csv(os.path.join(saving_path, name + '_' + from_paper + '_triple.dat'),
                   sep=' ',
                   header=False,
                   index=False)
    node_feature_df.to_csv(os.path.join(saving_path, name + '_' + from_paper + '_pre_train.emb'),
                           sep=' ',
                           header=False,
                           index=False)
    line_prepender(os.path.join(saving_path, name + '_' + from_paper + '_pre_train.emb'),
                   str(node_feature_df.shape[0]) + ' ' + str(node_feature_df.shape[1] - 1))
    return node_feature_df, triples


# #######################################
# Generic helper functions
# #######################################
def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)
