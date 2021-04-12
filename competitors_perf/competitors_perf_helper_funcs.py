import pandas as pd
import numpy as np
import os

from datasets import DBLP_ACM_IMDB_from_NSHE


def NSHE_dataset_to_edgelist(root='/home/ubuntu/msandal_code/PyG_playground/data/NSHE', name='dblp',
                             output_dir='/home/ubuntu/msandal_code/PyG_playground/competitors_perf/input_for_competitors'):
    """
    helper for the DeepWalk baseline: NSHE graphs  need to be transformed into
    edgelist files
    :param root: where to find/download NSHE dataset in question
    :param name: which dataset to fetch. must be one of ['dblp', 'imdb', 'acm']
    :param output_dir: where to save the resulting edge list
    :return:
    """
    assert str(name).lower() in ['dblp', 'acm', 'imdb'], \
        'NSHE_dataset_to_edgelist(): invalid dataset name required'
    ds = DBLP_ACM_IMDB_from_NSHE(root=root, name=str(name).lower())[0]
    edge_types = list(ds['edge_index_dict'].keys())
    edge_list = ds['edge_index_dict'][edge_types[0]].numpy().T
    for i in range(1, len(edge_types)):
        edge_list = np.vstack([edge_list, ds['edge_index_dict'][edge_types[i]].numpy().T])
    edge_df = pd.DataFrame(data=edge_list, columns=['id1', 'id2'])
    edge_df.to_csv(os.path.join(output_dir, str(name) + '_from_nshe_edgelist.txt'), sep = ' ',
                   header=False, index=False)
    return
