import pandas as pd
import numpy as np
import os

from datasets import DBLP_ACM_IMDB_from_NSHE, IMDB_ACM_DBLP_from_GTN


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
