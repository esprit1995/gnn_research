import torch
import os
import numpy as np

from datasets import IMDB_ACM_DBLP_from_GTN, DBLP_ACM_IMDB_from_NSHE
from torch_geometric.nn import MetaPath2Vec
from competitors_perf.baselines.baseline_data_transforms import metapath2vec_BDT
from downstream_tasks.evaluation_funcs import evaluate_clu_cla_GTN_NSHE_datasets
from utils.tools import ESim_parse_embeddings
from tqdm import tqdm
from termcolor import cprint


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
    emb_dim = model_init_params['emb_dim']
    walk_length = model_init_params['walk_length']
    context_size = model_init_params['context_size']
    walks_per_node = model_init_params['walks_per_node']
    num_neg_samples = model_init_params['num_neg_samples']

    edge_index_dict, ds, metapath = metapath2vec_BDT(args)
    model = MetaPath2Vec(edge_index_dict, embedding_dim=emb_dim,
                         metapath=metapath, walk_length=walk_length, context_size=context_size,
                         walks_per_node=walks_per_node, num_negative_samples=num_neg_samples,
                         sparse=True)
    loader = model.loader(batch_size=64, shuffle=True)
    optimizer = torch.optim.SparseAdam(params=list(model.parameters()),
                                       lr=args.lr)
    cprint('Training Metapath2Vec model on ' + str(args.dataset) + ' for ' + str(args.epochs) + ' epochs...',
           color='cyan')
    epoch_num = list()
    metrics = {'nmi': list(),
               'ari': list(),
               'macrof1': list(),
               'microf1': list()}
    for epoch in range(args.epochs):
        total_loss = 0
        if epoch == 0 or epoch % args.downstream_eval_freq == 0:
            model.eval()
            cprint('--->Evaluating downstream tasks...', color='yellow')
            epoch_num.append(epoch)
            nmi, ari, microf1, macrof1 = evaluate_clu_cla_GTN_NSHE_datasets(dataset=ds,
                                                                            embeddings=model.embedding.weight.data,
                                                                            verbose=False)
            metrics['nmi'].append(nmi)
            metrics['ari'].append(ari)
            metrics['microf1'].append(microf1)
            metrics['macrof1'].append(macrof1)
            print("this epoch's NMI : " + str(nmi))
            print("this epoch's ARI : " + str(ari))
            print('--> done!')
            cprint('--->Done!', color='yellow')
        model.train()
        print('Epoch ' + str(epoch + 1) + ' out of ' + str(args.epochs) + ' training: ')
        for pos_rw, neg_rw in tqdm(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('Epoch ' + str(epoch + 1) + ' loss: ' + str(total_loss / len(loader)))
    model.eval()
    return epoch_num, metrics


def evaluate_ESim_embeddings(emb_path='/home/ubuntu/msandal_code/PyG_playground/competitors_perf/competitor_embeddings'):
    """
    runs evaluation on ready node embeddings produced by ESim model
    (https://github.com/shangjingbo1226/ESim)
    :param emb_path: path to where the embeddings are stored
    :return:
    """
    datasets_to_evaluate = ['acm_gtn', 'acm_nshe', 'dblp_gtn', 'dblp_nshe']
    res_dict = {}
    for dataset in datasets_to_evaluate:
        filename = 'vec_' + dataset + '.txt'
        embeddings = np.fromfile(os.path.join(emb_path, filename), sep=' ')

        if dataset.split('_')[1] == 'gtn':
            name = str(dataset.split('_')[0]).upper()
            root = '/home/ubuntu/msandal_code/PyG_playground/data/IMDB_ACM_DBLP/' + name
            ds = IMDB_ACM_DBLP_from_GTN(root=root,
                                        name=name,
                                        initial_embs='original',
                                        redownload=False)[0]
            n_nodes = ds['node_type_mask'].shape[0]
        elif dataset.split('_')[1] == 'nshe':
            name = str(dataset.split('_')[0]).lower()
            root = '/home/ubuntu/msandal_code/PyG_playground/data/NSHE/' + name
            ds = DBLP_ACM_IMDB_from_NSHE(root=root,
                                         name=name,
                                         redownload=False)[0]
            n_nodes = ds['node_type_mask'].shape[0]
        else:
            raise ValueError('ESim eval: unexpected dataset to evaluate')
        embeddings = embeddings.reshape((n_nodes, -1)) # reshape into matrix
        embeddings = embeddings[np.argsort(embeddings[:, 0])] # sort in ascending order by node id (first column)
        embeddings = embeddings[:, :1] # drop node ids
        nmi, ari, microf1, macrof1 = evaluate_clu_cla_GTN_NSHE_datasets(dataset=ds,
                                                                        embeddings=embeddings,
                                                                        verbose=False)
        res_dict[dataset] = {'nmi': nmi, 'ari': ari, 'microf1': microf1, 'macrof1': macrof1}
    return res_dict

