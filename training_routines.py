import torch
import pandas as pd
from models import RGCN, GTN, HAN
from data_transforms import GTN_for_rgcn, GTN_for_gtn, ACM_HAN_for_han, NSHE_for_rgcn, NSHE_for_gtn
from utils.tools import heterogeneous_negative_sampling_naive, IMDB_DBLP_ACM_metapath_instance_sampler, \
    label_dict_to_metadata
from utils.losses import triplet_loss_pure, triplet_loss_type_aware, push_pull_metapath_instance_loss
from downstream_tasks.evaluation_funcs import evaluate_clu_cla_GTN_NSHE_datasets


def train_rgcn(args):
    # RGCN settings ##########
    output_dim = 64
    hidden_dim = 64
    num_layers = 2
    coclustering_metapaths_dict = {'ACM': [('0', '1', '0', '2'), ('2', '0', '1')],
                                   'DBLP': [('0', '1', '2'), ('0', '1', '0'), ('1', '2', '1')]}
    corruption_positions_dict = {'ACM': [(1, 2), (2, 2)],
                                 'DBLP': [(1, 1), (1, 2), (2, 2)]}
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
    metrics = list()
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
        if (epoch + 1) % args.downstream_eval_freq == 0:
            print('--> evaluating downstream tasks...')
            epoch_num.append(epoch+1)
            metrics.append(evaluate_clu_cla_GTN_NSHE_datasets(dataset=ds, embeddings=output, verbose=False)[0])
            print('--> done!')
        if epoch % 5 == 0:
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
    coclustering_metapaths_dict = {'ACM': [('0', '1', '0', '2'), ('2', '0', '1')],
                                   'DBLP': [('0', '1', '2'), ('0', '1', '0'), ('1', '2', '1')]}
    corruption_positions_dict = {'ACM': [(1, 2), (2, 2)],
                                 'DBLP': [(1, 1), (1, 2), (2, 2)]}
    setattr(args, 'model_params', model_params)
    # #######################
    # ---> get necessary data structures
    if args.from_paper == 'GTN':
        A, node_label_dict, node_features, num_classes, edge_index, edge_type, id_type_mask, ds = GTN_for_gtn(
            name=args.dataset)
    elif args.from_paper == 'NSHE':
        A, node_label_dict, node_features, num_classes, edge_index, edge_type, id_type_mask, ds = NSHE_for_gtn(
            name=args.dataset)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
    # keeping track of performance vs #epochs
    epoch_num = list()
    metrics = list()
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
            epoch_num.append(epoch+1)
            metrics.append(evaluate_clu_cla_GTN_NSHE_datasets(dataset=ds, embeddings=output, verbose=False)[0])
            print('--> done!')
        print("Epoch: ", epoch, " loss: ", loss)

    all_ids, all_labels = label_dict_to_metadata(node_label_dict)
    return output, all_ids, all_labels, id_type_mask, ds, epoch_num, metrics


def train_han(args):
    # HAN settings ##########
    num_heads = 3,
    dropout = 0
    out_size = 20
    hidden_size = 40
    metapaths = [('pa', 'ap'), ('pf', 'fp')]
    model_params = {'num_heads': num_heads, 'dropout': dropout,
                    'outsize': out_size, 'hidden_size': hidden_size, 'metapaths': metapaths}
    # #######################
    setattr(args, 'model_params', model_params)
    if args.dataset == 'ACM_HAN':
        g, features, labels, num_classes, edge_index_list, id_type_mask = ACM_HAN_for_han()
    else:
        raise ValueError("Currently HAN cannot be trained for dataset: ", args.dataset)

    model = HAN(meta_paths=metapaths,
                in_size=features.shape[1],
                hidden_size=hidden_size,
                out_size=out_size,
                num_heads=num_heads,
                dropout=dropout).to(args.device)
    g = g.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    model.train()
    for epoch in range(args.epochs):
        output = model(g, features)
        meta_triplets = [heterogeneous_negative_sampling_naive(edge_index, id_type_mask)
                         for edge_index in edge_index_list]
        triplets = tuple([torch.cat([meta_triplets[i][j] for i in range(len(meta_triplets))]) for j in range(3)])
        loss = triplet_loss_type_aware(triplets, output, id_type_mask, lmbd=args.type_lambda) if args.type_aware_loss \
            else triplet_loss_pure(triplets, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch: ", epoch, " loss: ", loss)

    return output, id_type_mask
