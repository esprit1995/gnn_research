from torch_geometric.nn import MetaPath2Vec
from baseline_data_transforms import metapath2vec_BDT

def train_metapath2vec(args,
                       model_init_params: dict = {'emb_dim': 64,
                                                  'walk_length': 20,
                                                  'context_size': 10,
                                                  'walks_per_node': 5,
                                                  'num_neg_samples': 5}):
    """
    train the metapath2vec model to get a baseline co-clustering performance
    :param args: experiment arguments
    :param model_init_params: initial parameters for metapath2vec, optional
    :return:
    """
    # Metapath2vec settings
    metapath = [('user', 'likes', 'meme'),
                ('meme', 'liked by', 'user'),
                ('user', 'likes', 'meme'),
                ('meme', 'disliked by', 'user'),
                ('user', 'dislikes', 'meme'),
                ('meme', 'disliked by', 'user')]
    emb_dim = model_init_params['emb_dim']
    walk_length = model_init_params['walk_length']
    context_size = model_init_params['context_size']
    walks_per_node = model_init_params['walks_per_node']
    num_neg_samples = model_init_params['num_neg_samples']

    edge_index_dict, ds, metapath  = metapath2vec_BDT(args)

