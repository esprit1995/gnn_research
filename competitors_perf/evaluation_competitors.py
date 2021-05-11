import numpy as np
import os
from downstream_tasks.evaluation_funcs import evaluate_clu_cla_GTN_NSHE_datasets
from datasets import DBLP_ACM_IMDB_from_NSHE, IMDB_ACM_DBLP_from_GTN

DEFAULT_COMP_EMB_PATH = '/home/ubuntu/msandal_code/PyG_playground/competitors_perf/competitor_embeddings'
DEFAULT_DATA_PATH = '/home/ubuntu/msandal_code/PyG_playground/data/'


def evaluate_competitor(path_to_embs: str = DEFAULT_COMP_EMB_PATH,
                        dataset: str = 'dblp',
                        from_paper: str = 'nshe',
                        evaluate_architecture: str = 'nshe',
                        with_cocluster_loss: bool=False):
    """
    evaluate clustering/classification on a chosen dataset using
    embeddings produced by different GNN architectures
    :param path_to_embs: where the embeddings are stored
    :param dataset: which dataset to evaluate. Can be: ['dblp', 'imdb', 'acm']
    :param from_paper: from which paper the dataset version comes. Can be: ['nshe', 'gtn']
    :param evaluate_architecture: which architecture to evaluate. Possible values: ['deepwalk', 'nshe', 'hegan_gen', 'hegan_dis']
    :param with_cocluster_loss: whether to use the embeddings obtained with coclustering complimentary loss
    :return:
    """
    # === prepare and validate the arguments
    dataset = str(dataset).lower()
    from_paper = str(from_paper).lower()
    evaluate_architecture = str(evaluate_architecture).lower()
    assert dataset in ['dblp', 'acm', 'imdb'], \
        'evaluate_competitor(): invalid dataset requested'
    assert from_paper in ['nshe', 'gtn'], \
        'evaluate_competitor(): invalid from_paper argument'
    assert evaluate_architecture in ['nshe', 'deepwalk', 'hegan_gen', 'hegan_dis'], \
        'evaluate_competitor(): cannot evaluate given architecture: ' + str(evaluate_architecture)

    # === get embeddings
    if evaluate_architecture == 'deepwalk':
        emb_filename = dataset + "_from_" + from_paper + "_deepwalk.embeddings"
        with open(os.path.join(path_to_embs, emb_filename)) as f:
            lines = (line for line in f)
            embs = np.loadtxt(lines, delimiter=' ', skiprows=1)
        embs = embs[np.argsort(embs[:, 0])]  # sort by node id, which is the first column
        embs = embs[:, 1:]  # remove the first column, as it is not a feature
    elif evaluate_architecture in ['hegan_dis', 'hegan_gen']:
        if with_cocluster_loss:
            emb_filename = dataset + "_from_" + from_paper + "_" + evaluate_architecture + '_ccl.emb'
        else:
            emb_filename = dataset + "_from_" + from_paper + "_" + evaluate_architecture + '.emb'
        with open(os.path.join(path_to_embs, emb_filename)) as f:
            lines = (line for line in f)
            embs = np.loadtxt(lines, delimiter=' ', skiprows=1)
        embs = embs[np.argsort(embs[:, 0])]  # sort by node id, which is the first column
        embs = embs[:, 1:]  # remove the first column, as it is not a feature
    elif evaluate_architecture == 'nshe':
        emb_filename = dataset + "_from_" + from_paper + "_nshe_embeddings.npy"
        embs = np.load(os.path.join(path_to_embs, emb_filename))
    else:
        raise NotImplementedError('evaluate_competitor(): evaluation not supported for architecture: ' + str(evaluate_architecture))

    # === fetch PyG dataset which we evaluate
    if from_paper == 'nshe':
        dataset = DBLP_ACM_IMDB_from_NSHE(root=os.path.join(DEFAULT_DATA_PATH, 'NSHE'),
                                          name=dataset.lower())[0]
    elif from_paper == 'gtn':
        dataset = IMDB_ACM_DBLP_from_GTN(root=os.path.join(DEFAULT_DATA_PATH, 'IMDB_ACM_DBLP', dataset.upper()),
                                         name=dataset.upper())[0]

    # === evaluation
    evaluate_clu_cla_GTN_NSHE_datasets(dataset, embs)
