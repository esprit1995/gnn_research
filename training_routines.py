import torch
import numpy as np
import pandas as pd

from torch_geometric.nn import VGAE
from models import RGCN, GTN, NSHE, MAGNN, HeGAN, VariationalRGCNEncoder
from data_transforms import GTN_for_rgcn, GTN_for_gtn, NSHE_for_rgcn, NSHE_for_gtn, GTN_or_NSHE_for_nshe, \
    GTN_NSHE_for_MAGNN, GTN_NSHE_for_HeGAN
from utils.tools import heterogeneous_negative_sampling_naive, IMDB_DBLP_ACM_metapath_instance_sampler, \
    label_dict_to_metadata
from utils.losses import triplet_loss_pure, push_pull_metapath_instance_loss, NSHE_network_schema_loss
from downstream_tasks.evaluation_funcs import evaluate_clu_cla_GTN_NSHE_datasets, \
    evaluate_link_prediction_GTN_NSHE_datasets
from utils.MAGNN_utils import nega_sampling, Batcher, prepare_minibatch

# a collection of shorter metapaths
# COCLUSTERING_METAPATHS = {'ACM': [('0', '1', '0', '2'), ('2', '0', '1')],
#                           'DBLP': [('0', '1', '2'), ('0', '1', '0'), ('1', '2', '1')]}
# CORRUPTION_POSITIONS = {'ACM': [(1, 2), (2, 2)],
#                         'DBLP': [(1, 1), (1, 2), (2, 2)]}

# single long metapath
# COCLUSTERING_METAPATHS = {'ACM': [('1', '0', '2', '0', '1')],
#                           'DBLP': [('0', '1', '2', '1', '0')]}
# CORRUPTION_POSITIONS = {'ACM': [(1, 3)],
#                         'DBLP': [(2, 4)]}

# mix of the above
COCLUSTERING_METAPATHS = {'ACM': [('0', '1', '0', '2'), ('2', '0', '1'), ('1', '0', '2', '0', '1')],
                          'DBLP': [('0', '1', '2'), ('0', '1', '0'), ('1', '2', '1'), ('0', '1', '2', '1', '0')]}
CORRUPTION_POSITIONS = {'ACM': [(1, 2), (2, 2), (1, 3)],
                        'DBLP': [(1, 1), (1, 2), (2, 2), (2, 4)]}


def train_rgcn(args):
    # RGCN settings ##########
    output_dim = 64
    hidden_dim = 64
    num_layers = 2
    coclustering_metapaths_dict = COCLUSTERING_METAPATHS
    corruption_positions_dict = CORRUPTION_POSITIONS
    # #######################

    # ========> preparing data: wrangling, sampling
    if args.from_paper == 'GTN':
        ds, n_node_dict, node_label_dict, id_type_mask, node_feature_matrix, edge_index, edge_type = GTN_for_rgcn(
            args.dataset, args)
    elif args.from_paper == 'NSHE':
        ds, n_node_dict, node_label_dict, id_type_mask, node_feature_matrix, edge_index, edge_type = NSHE_for_rgcn(
            name=args.dataset, args=args, data_dir='/home/ubuntu/msandal_code/PyG_playground/data/NSHE')
    else:
        raise ValueError(
            'train_rgcn(): for requested paper ---' + str(args.from_paper) + '--- training RGCN is not possible')
    # sampling metapath instances for the cocluster push-pull loss
    if args.cocluster_loss:
        pos_instances = dict()
        neg_instances = dict()

        try:
            metapath_templates, corruption_positions = coclustering_metapaths_dict[args.dataset], \
                                                       corruption_positions_dict[
                                                           args.dataset]
        except Exception as e:
            raise ValueError('co-clustering loss is not supported for dataset name ' + str(args.dataset))

        for mptemplate_idx in range(len(metapath_templates)):
            pos_instances[metapath_templates[mptemplate_idx]], \
            neg_instances[metapath_templates[mptemplate_idx]] = IMDB_DBLP_ACM_metapath_instance_sampler(
                dataset=ds,
                metapath=metapath_templates[mptemplate_idx],
                n=args.instances_per_template,
                corruption_method=args.corruption_method,
                corruption_position=corruption_positions[mptemplate_idx])
    print('Data ready!')

    # ========> training RGCN
    model = RGCN(input_dim=node_feature_matrix.shape[1],
                 output_dim=output_dim,
                 hidden_dim=hidden_dim,
                 num_relations=pd.Series(edge_type.numpy()).nunique(),
                 num_layers=num_layers)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.001)
    losses = []
    # keeping track of performance vs #epochs
    epoch_num = list()
    metrics = {'nmi': list(),
               'ari': list(),
               'macrof1': list(),
               'microf1': list(),
               'roc_auc': list(),
               'f1': list()}

    for epoch in range(args.epochs):
        model = model.float()
        output = model(x=node_feature_matrix.float(),
                       edge_index=edge_index,
                       edge_type=edge_type)
        triplets = heterogeneous_negative_sampling_naive(edge_index, id_type_mask)

        # regular loss: triplet loss
        loss = triplet_loss_pure(triplets, output) if args.base_triplet_loss else 0
        # additional co-clustering losses
        if args.cocluster_loss:
            mptemplates = list(pos_instances.keys())
            for idx in range(len(mptemplates)):
                loss = loss + args.type_lambda * push_pull_metapath_instance_loss(pos_instances[mptemplates[idx]],
                                                                                  neg_instances[mptemplates[idx]],
                                                                                  corruption_positions[idx],
                                                                                  output)

        losses.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % args.downstream_eval_freq == 0 or epoch == 0:
            print('--> evaluating downstream tasks...')
            epoch_num.append(epoch + 1)
            nmi, ari, microf1, macrof1 = evaluate_clu_cla_GTN_NSHE_datasets(dataset=ds, embeddings=output,
                                                                            verbose=False)
            roc_auc, f1 = evaluate_link_prediction_GTN_NSHE_datasets(dataset=ds, embeddings=output, verbose=False)
            metrics['nmi'].append(nmi)
            metrics['ari'].append(ari)
            metrics['microf1'].append(microf1)
            metrics['macrof1'].append(macrof1)
            metrics['roc_auc'].append(roc_auc)
            metrics['f1'].append(f1)
            print("this epoch's NMI : " + str(nmi))
            print("this epoch's ARI : " + str(ari))
            print('--> done!')
        if epoch % 5 == 0:
            print("Epoch: ", epoch, " loss: ", loss)
    all_ids, all_labels = label_dict_to_metadata(node_label_dict)
    return output, all_ids, all_labels, id_type_mask, ds, epoch_num, metrics


def train_vgae(args):
    # VGAE settings ##########
    output_dim = 64
    hidden_dim = 64
    coclustering_metapaths_dict = COCLUSTERING_METAPATHS
    corruption_positions_dict = CORRUPTION_POSITIONS
    # #######################
    # ========> preparing data: wrangling, sampling
    if args.from_paper == 'GTN':
        ds, n_node_dict, node_label_dict, id_type_mask, node_feature_matrix, edge_index, edge_type = GTN_for_rgcn(
            args.dataset, args)
    elif args.from_paper == 'NSHE':
        ds, n_node_dict, node_label_dict, id_type_mask, node_feature_matrix, edge_index, edge_type = NSHE_for_rgcn(
            name=args.dataset, args=args, data_dir='/home/ubuntu/msandal_code/PyG_playground/data/NSHE')
    else:
        raise ValueError(
            'train_rgcn(): for requested paper ---' + str(args.from_paper) + '--- training RGCN is not possible')
    # sampling metapath instances for the cocluster push-pull loss
    if args.cocluster_loss:
        pos_instances = dict()
        neg_instances = dict()

        try:
            metapath_templates, corruption_positions = coclustering_metapaths_dict[args.dataset], \
                                                       corruption_positions_dict[
                                                           args.dataset]
        except Exception as e:
            raise ValueError('co-clustering loss is not supported for dataset name ' + str(args.dataset))

        for mptemplate_idx in range(len(metapath_templates)):
            pos_instances[metapath_templates[mptemplate_idx]], \
            neg_instances[metapath_templates[mptemplate_idx]] = IMDB_DBLP_ACM_metapath_instance_sampler(
                dataset=ds,
                metapath=metapath_templates[mptemplate_idx],
                n=args.instances_per_template,
                corruption_method=args.corruption_method,
                corruption_position=corruption_positions[mptemplate_idx])
    print('Data ready!')

    model = VGAE(VariationalRGCNEncoder(node_feature_matrix.shape[1],
                                        output_dim,
                                        num_relations=pd.Series(edge_type.numpy()).nunique()))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=args.weight_decay)
    # keeping track of performance vs #epochs
    epoch_num = list()
    metrics = {'nmi': list(),
               'ari': list(),
               'macrof1': list(),
               'microf1': list(),
               'roc_auc': list(),
               'f1': list()}
    output = None
    for epoch in range(args.epochs):
        model.zero_grad()
        output = model.encode(x=node_feature_matrix.float(),
                              edge_index=edge_index,
                              edge_type=edge_type)
        # regular loss: triplet loss
        loss = model.recon_loss(output, edge_index) + (1 / id_type_mask.shape[0]) * model.kl_loss()
        # additional co-clustering losses
        if args.cocluster_loss:
            mptemplates = list(pos_instances.keys())
            for idx in range(len(mptemplates)):
                loss = loss + args.type_lambda * push_pull_metapath_instance_loss(pos_instances[mptemplates[idx]],
                                                                                  neg_instances[mptemplates[idx]],
                                                                                  corruption_positions[idx],
                                                                                  output)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % args.downstream_eval_freq == 0 or epoch == 0:
            print('--> evaluating downstream tasks...')
            epoch_num.append(epoch + 1)
            nmi, ari, microf1, macrof1 = evaluate_clu_cla_GTN_NSHE_datasets(dataset=ds, embeddings=output,
                                                                            verbose=False)
            roc_auc, f1 = evaluate_link_prediction_GTN_NSHE_datasets(dataset=ds, embeddings=output, verbose=False)
            metrics['nmi'].append(nmi)
            metrics['ari'].append(ari)
            metrics['microf1'].append(microf1)
            metrics['macrof1'].append(macrof1)
            metrics['roc_auc'].append(roc_auc)
            metrics['f1'].append(f1)
            print("this epoch's NMI : " + str(nmi))
            print("this epoch's ARI : " + str(ari))
            print('--> done!')
        print("Epoch: ", epoch, " loss: ", loss)

    all_ids, all_labels = label_dict_to_metadata(node_label_dict)
    return output, all_ids, all_labels, id_type_mask, ds, epoch_num, metrics


def train_gtn(args):
    # GTN settings ##########
    node_dim = 32
    num_channels = 2
    num_layers = 2
    norm = 'true'
    model_params = {'node_dim': node_dim, 'num_channels': num_channels,
                    'num_layers': num_layers, 'norm': norm}
    coclustering_metapaths_dict = COCLUSTERING_METAPATHS
    corruption_positions_dict = CORRUPTION_POSITIONS
    setattr(args, 'model_params', model_params)
    # #######################
    # ---> get necessary data structures
    if args.from_paper == 'GTN':
        A, node_label_dict, node_features, num_classes, edge_index, edge_type, id_type_mask, ds = GTN_for_gtn(
            args)
    elif args.from_paper == 'NSHE':
        A, node_label_dict, node_features, num_classes, edge_index, edge_type, id_type_mask, ds = NSHE_for_gtn(
            args)
    else:
        raise NotImplementedError('GTN cannot be trained on datasets from paper: ' + str(args.from_paper))
    node_features = node_features.float()
    A = A.float()
    # ---> additional co-clustering loss data structures
    if args.cocluster_loss:
        pos_instances = dict()
        neg_instances = dict()

        try:
            metapath_templates, corruption_positions = coclustering_metapaths_dict[args.dataset], \
                                                       corruption_positions_dict[
                                                           args.dataset]
        except Exception as e:
            raise ValueError('co-clustering loss is not supported for dataset name ' + str(args.dataset))

        for mptemplate_idx in range(len(metapath_templates)):
            pos_instances[metapath_templates[mptemplate_idx]], \
            neg_instances[metapath_templates[mptemplate_idx]] = IMDB_DBLP_ACM_metapath_instance_sampler(
                dataset=ds,
                metapath=metapath_templates[mptemplate_idx],
                n=args.instances_per_template,
                corruption_method=args.corruption_method,
                corruption_position=corruption_positions[mptemplate_idx])

    model = GTN(num_edge=A.shape[-1],
                num_channels=num_channels,
                w_in=node_features.shape[1],
                w_out=node_dim,
                num_class=num_classes,
                num_layers=num_layers,
                norm=norm)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=args.weight_decay)
    # keeping track of performance vs #epochs
    epoch_num = list()
    metrics = {'nmi': list(),
               'ari': list(),
               'macrof1': list(),
               'microf1': list(),
               'roc_auc': list(),
               'f1': list()}
    output = None
    for epoch in range(args.epochs):
        model.zero_grad()
        output = model(A, node_features)
        triplets = heterogeneous_negative_sampling_naive(edge_index, id_type_mask)
        # regular loss: triplet loss
        loss = triplet_loss_pure(triplets, output) if args.base_triplet_loss else 0
        # additional co-clustering losses
        if args.cocluster_loss:
            mptemplates = list(pos_instances.keys())
            for idx in range(len(mptemplates)):
                loss = loss + args.type_lambda * push_pull_metapath_instance_loss(pos_instances[mptemplates[idx]],
                                                                                  neg_instances[mptemplates[idx]],
                                                                                  corruption_positions[idx],
                                                                                  output)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % args.downstream_eval_freq == 0 or epoch == 0:
            print('--> evaluating downstream tasks...')
            epoch_num.append(epoch + 1)
            nmi, ari, microf1, macrof1 = evaluate_clu_cla_GTN_NSHE_datasets(dataset=ds, embeddings=output,
                                                                            verbose=False)
            roc_auc, f1 = evaluate_link_prediction_GTN_NSHE_datasets(dataset=ds, embeddings=output, verbose=False)
            metrics['nmi'].append(nmi)
            metrics['ari'].append(ari)
            metrics['microf1'].append(microf1)
            metrics['macrof1'].append(macrof1)
            metrics['roc_auc'].append(roc_auc)
            metrics['f1'].append(f1)
            print("this epoch's NMI : " + str(nmi))
            print("this epoch's ARI : " + str(ari))
            print('--> done!')
        print("Epoch: ", epoch, " loss: ", loss)

    all_ids, all_labels = label_dict_to_metadata(node_label_dict)
    return output, all_ids, all_labels, id_type_mask, ds, epoch_num, metrics


def train_nshe(args):
    # dummy class for NSHE params
    class Hyperparameters(object):
        pass

    # NSHE settings ##########
    hp = Hyperparameters()
    optim_beta = {'acm': 33.115,
                  'dblp': 0.905,
                  'imdb': 0.05}
    hp.conv_method = 'GCNx1'
    hp.cla_layers = 2
    hp.ns_emb_mode = 'TypeSpecCla'
    hp.cla_method = 'TypeSpecCla'
    hp.norm_emb_flag = True
    hp.size = {'com_feat_dim': 128, 'emb_dim': 128}
    hp.beta = optim_beta[str(args.dataset).lower()]
    model_params = {'conv_method': hp.conv_method, 'cla_layers': hp.cla_layers,
                    'ns_emb_mode': hp.ns_emb_mode, 'cla_method': hp.cla_method,
                    'norm_emb_flag': hp.norm_emb_flag, 'size': hp.size, 'beta': hp.beta}
    coclustering_metapaths_dict = COCLUSTERING_METAPATHS
    corruption_positions_dict = CORRUPTION_POSITIONS
    setattr(args, 'model_params', model_params)
    # #######################
    # ---> get necessary data structures
    if args.from_paper == 'GTN':
        g = GTN_or_NSHE_for_nshe(args=args, data_dir='/home/ubuntu/msandal_code/PyG_playground/data/IMDB_ACM_DBLP')
    elif args.from_paper == 'NSHE':
        g = GTN_or_NSHE_for_nshe(args=args, data_dir='/home/ubuntu/msandal_code/PyG_playground/data/NSHE/')
    else:
        raise NotImplementedError('NSHE cannot be trained on datasets from paper: ' + str(args.from_paper))

    # ---> additional co-clustering loss data structures
    pos_instances = None
    neg_instances = None
    if args.cocluster_loss:
        pos_instances = dict()
        neg_instances = dict()

        try:
            metapath_templates, corruption_positions = coclustering_metapaths_dict[args.dataset], \
                                                       corruption_positions_dict[
                                                           args.dataset]
        except Exception as e:
            raise ValueError('co-clustering loss is not supported for dataset name ' + str(args.dataset) +
                             '\n' + str(e))

        for mptemplate_idx in range(len(metapath_templates)):
            pos_instances[metapath_templates[mptemplate_idx]], \
            neg_instances[metapath_templates[mptemplate_idx]] = IMDB_DBLP_ACM_metapath_instance_sampler(
                dataset=g.ds,
                metapath=metapath_templates[mptemplate_idx],
                n=args.instances_per_template,
                corruption_method=args.corruption_method,
                corruption_position=corruption_positions[mptemplate_idx])

    # ---> instantiate model and optimizer
    model = NSHE(g, hp)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # keeping track of performance vs #epochs
    epoch_num = list()
    metrics = {'nmi': list(),
               'ari': list(),
               'macrof1': list(),
               'microf1': list(),
               'roc_auc': list(),
               'f1': list()}
    output = None
    for epoch in range(args.epochs):
        model.zero_grad()
        g.get_epoch_samples(epoch, args)
        output, schema_preds = model(g.adj, g.feature, g.ns_instances)
        triplets = heterogeneous_negative_sampling_naive(g.edge_index, g.id_type_mask)
        # regular loss: pairwise proximity loss. We use our loss, but it is very similar to the one
        # used in the original paper
        loss = triplet_loss_pure(triplets, output) if args.base_triplet_loss else 0

        # network schema-preserving loss (multiplied by a coeff taken from the original paper)
        loss = loss + hp.beta * NSHE_network_schema_loss(schema_preds, g.ns_label)

        # additional co-clustering loss
        if args.cocluster_loss:
            mptemplates = list(pos_instances.keys())
            for idx in range(len(mptemplates)):
                loss = loss + args.type_lambda * push_pull_metapath_instance_loss(pos_instances[mptemplates[idx]],
                                                                                  neg_instances[mptemplates[idx]],
                                                                                  corruption_positions[idx],
                                                                                  output)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % args.downstream_eval_freq == 0 or epoch == 0:
            print('--> evaluating downstream tasks...')
            epoch_num.append(epoch + 1)
            nmi, ari, microf1, macrof1 = evaluate_clu_cla_GTN_NSHE_datasets(dataset=g.ds, embeddings=output,
                                                                            verbose=False)
            roc_auc, f1 = evaluate_link_prediction_GTN_NSHE_datasets(dataset=g.ds, embeddings=output, verbose=False)
            metrics['nmi'].append(nmi)
            metrics['ari'].append(ari)
            metrics['microf1'].append(microf1)
            metrics['macrof1'].append(macrof1)
            metrics['roc_auc'].append(roc_auc)
            metrics['f1'].append(f1)
            print("this epoch's NMI : " + str(nmi))
            print("this epoch's ARI : " + str(ari))
            print('--> done!')
        print("Epoch: ", epoch, " loss: ", loss)

    all_ids, all_labels = label_dict_to_metadata(g.node_label_dict)
    return output, all_ids, all_labels, g.id_type_mask, g.ds, epoch_num, metrics


def train_magnn(args):
    # MAGNN settings ##########
    hdim = 50
    adim = 100
    dropout = 0.5
    device = args.device
    nlayer = 2
    nlabel = 0
    nhead = 8
    rtype = 'RotatE0'
    sampling = 100
    model_params = {'hdim': hdim, 'adim': adim,
                    'dropout': dropout, 'nlayer': nlayer,
                    'nlabel': nlabel, 'nhead': nhead,
                    'rtype': rtype, 'sampling': sampling}
    setattr(args, 'model_params', model_params)

    coclustering_metapaths_dict = COCLUSTERING_METAPATHS
    corruption_positions_dict = CORRUPTION_POSITIONS
    # #######################
    # ---> get necessary data structures
    if args.from_paper in ['NSHE', 'GTN']:
        data_directory = '/home/ubuntu/msandal_code/PyG_playground/data/IMDB_ACM_DBLP' if args.from_paper == 'GTN' \
            else '/home/ubuntu/msandal_code/PyG_playground/data/NSHE'
        graph_statistics, type_mask, node_labels, node_order, ntype_features, posi_edges, node_mptype_mpinstances, \
        node_label_dict, id_type_mask, edge_index, ds = GTN_NSHE_for_MAGNN(args, data_dir=data_directory)
    else:
        raise ValueError('MAGNN cannot be trained on datasets from paper: ' + str(args.from_paper))

    # ---> additional co-clustering loss data structures
    if args.cocluster_loss:
        pos_instances = dict()
        neg_instances = dict()

        try:
            metapath_templates, corruption_positions = coclustering_metapaths_dict[args.dataset], \
                                                       corruption_positions_dict[
                                                           args.dataset]
        except Exception as e:
            raise ValueError('co-clustering loss is not supported for dataset name ' + str(args.dataset))

        for mptemplate_idx in range(len(metapath_templates)):
            pos_instances[metapath_templates[mptemplate_idx]], \
            neg_instances[metapath_templates[mptemplate_idx]] = IMDB_DBLP_ACM_metapath_instance_sampler(
                dataset=ds,
                metapath=metapath_templates[mptemplate_idx],
                n=args.instances_per_template,
                corruption_method=args.corruption_method,
                corruption_position=corruption_positions[mptemplate_idx])

    model = MAGNN(graph_statistics, hdim, adim, dropout, device, nlayer, nhead, nlabel,
                  ntype_features, rtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()

    # keeping track of performance vs #epochs
    epoch_num = list()
    metrics = {'nmi': list(),
               'ari': list(),
               'macrof1': list(),
               'microf1': list(),
               'roc_auc': list(),
               'f1': list()}
    output = None
    for epoch in range(args.epochs):
        #  batcher code is conserved in for the sake of code re-usage
        #  batcher code is corrected to perform full-batch training, hence batcher is
        #  re-instantiated on every epoch
        if epoch == 0:
            # evaluate the downstream tasks on the initial embeddings
            print('Evaluating initial embeddings...')
            init_feats = np.vstack([ntype_features[0], ntype_features[1], ntype_features[2]])
            NMI_init = evaluate_clu_cla_GTN_NSHE_datasets(dataset=ds, embeddings=init_feats, verbose=False)[0]
            print('NMI on the initial embeddings: ' + str(NMI_init))
        nega_edges = nega_sampling(len(type_mask), posi_edges)
        batcher = Batcher(False, 1, [posi_edges, nega_edges])
        batch_targets, batch_labels = batcher.next()
        layer_ntype_mptype_g, layer_ntype_mptype_mpinstances, layer_ntype_mptype_iftargets, batch_ntype_orders = prepare_minibatch(
            set(batch_targets.flatten()), node_mptype_mpinstances, type_mask, node_order, nlayer, sampling,
            args.device)

        batch_node_features, _ = model(layer_ntype_mptype_g, layer_ntype_mptype_mpinstances,
                                       layer_ntype_mptype_iftargets, batch_ntype_orders)
        indices = list(batch_node_features.keys())
        assert all(indices[i] < indices[i + 1] for i in range(len(indices) - 1)), \
            "MAGNN: indices are not sorted, aborting."
        output = torch.vstack([batch_node_features[key] for key in list(batch_node_features.keys())])

        triplets = heterogeneous_negative_sampling_naive(edge_index, id_type_mask)
        # regular loss: triplet loss
        loss = triplet_loss_pure(triplets, output) if args.base_triplet_loss else 0
        # additional co-clustering losses
        if args.cocluster_loss:
            mptemplates = list(pos_instances.keys())
            for idx in range(len(mptemplates)):
                loss = loss + args.type_lambda * push_pull_metapath_instance_loss(pos_instances[mptemplates[idx]],
                                                                                  neg_instances[mptemplates[idx]],
                                                                                  corruption_positions[idx],
                                                                                  output)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        del batch_node_features, _
        print("Epoch: ", epoch, " loss: ", loss)
        if (epoch + 1) % args.downstream_eval_freq == 0 or epoch == 0:
            print('--> evaluating downstream tasks...')
            epoch_num.append(epoch + 1)
            nmi, ari, microf1, macrof1 = evaluate_clu_cla_GTN_NSHE_datasets(dataset=ds, embeddings=output,
                                                                            verbose=False)
            roc_auc, f1 = evaluate_link_prediction_GTN_NSHE_datasets(dataset=ds, embeddings=output, verbose=False)
            metrics['nmi'].append(nmi)
            metrics['ari'].append(ari)
            metrics['microf1'].append(microf1)
            metrics['macrof1'].append(macrof1)
            metrics['roc_auc'].append(roc_auc)
            metrics['f1'].append(f1)
            print("this epoch's NMI : " + str(nmi))
            print("this epoch's ARI : " + str(ari))
            print('--> done!')

    all_ids, all_labels = label_dict_to_metadata(node_label_dict)
    return output, all_ids, all_labels, id_type_mask, ds, epoch_num, metrics


def train_hegan(args):
    # HeGAN settings ##########
    batch_size = 512
    lambda_gen = 1e-5
    lambda_dis = 1e-5
    n_sample = 16
    lr_gen = 0.0001  # 1e-3
    lr_dis = 0.0001  # 1e-4
    n_epoch = args.epochs
    saves_step = 10
    sig = 1.0
    d_epoch = 15
    g_epoch = 5
    hidden_dim = 128
    model_params = {'batch_size': batch_size, 'lambda_gen': lambda_gen, 'lambda_dis': lambda_dis,
                    'n_sample': n_sample, 'lr_gen': lr_gen, 'lr_dis': lr_dis,
                    'n_epoch': n_epoch, 'saves_step': saves_step, 'sig': sig,
                    'd_epoch': d_epoch, 'g_epoch': g_epoch, 'hidden_dim': hidden_dim}
    coclustering_metapaths_dict = COCLUSTERING_METAPATHS
    corruption_positions_dict = CORRUPTION_POSITIONS
    setattr(args, 'model_params', model_params)
    # #######################
    # ---> get necessary data structures
    if args.from_paper in ['NSHE', 'GTN']:
        data_directory = '/home/ubuntu/msandal_code/PyG_playground/data/IMDB_ACM_DBLP' if args.from_paper == 'GTN' \
            else '/home/ubuntu/msandal_code/PyG_playground/data/NSHE'
        config, node_label_dict, id_type_mask, edge_index, ds = GTN_NSHE_for_HeGAN(args=args,
                                                                                   data_dir=data_directory,
                                                                                   model_params=model_params)
    else:
        raise ValueError('HeGAN cannot be trained on datasets from paper: ' + str(args.from_paper))

    # ---> additional co-clustering loss data structures
    if args.cocluster_loss:
        pos_instances = dict()
        neg_instances = dict()

        try:
            metapath_templates, corruption_positions = coclustering_metapaths_dict[args.dataset], \
                                                       corruption_positions_dict[
                                                           args.dataset]
        except Exception as e:
            raise ValueError('co-clustering loss is not supported for dataset name ' + str(args.dataset))

        for mptemplate_idx in range(len(metapath_templates)):
            pos_instances[metapath_templates[mptemplate_idx]], \
            neg_instances[metapath_templates[mptemplate_idx]] = IMDB_DBLP_ACM_metapath_instance_sampler(
                dataset=ds,
                metapath=metapath_templates[mptemplate_idx],
                n=args.instances_per_template,
                corruption_method=args.corruption_method,
                corruption_position=corruption_positions[mptemplate_idx])
    else:
        pos_instances = None
        neg_instances = None
        corruption_positions = None

    ccl_loss_structures = {'pos_instances': pos_instances,
                           'neg_instances': neg_instances,
                           'corruption_positions': corruption_positions,
                           'coclustering_metapaths_dict': coclustering_metapaths_dict,
                           'corruption_positions_dict': corruption_positions_dict}
    model = HeGAN(args, config, ds, ccl_loss_structures)
    model.train()
    print('HeGAN trained')
    output = torch.tensor(model.output_embs_dis)
    all_ids, all_labels = label_dict_to_metadata(node_label_dict)
    return output, all_ids, all_labels, id_type_mask, ds, model.epoch_num, model.metrics
