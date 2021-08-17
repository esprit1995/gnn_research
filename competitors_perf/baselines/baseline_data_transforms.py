from datasets import *


def edge_index_dict_individual_ids(edge_index_dict: dict, node_type_mask: torch.tensor):
    """
    make ids of each type start with 0
    :param edge_index_dict: edge_index_dict with keys of type ('int_1', 'int_2')
    :param node_type_mask: node type mask for the given graph
                           node_types are expected to be ascendingly sorted (e.g. [0, ..., 0, 1, ..., 1, 2, ...])
    :return: transformed edge_index_dict
    """
    nnodes_dict = {str(val.item()): len([elem for elem in node_type_mask if elem == val]) for val in
                   node_type_mask.unique()}
    nnodes_list = [nnodes_dict[key] for key in list(nnodes_dict.keys())]

    offsets_dict = {'0': 0}
    for key in list(nnodes_dict.keys()):
        if key == '0':
            continue
        else:
            offsets_dict[key] = sum(nnodes_list[:list(nnodes_dict.keys()).index(key)])

    transformed_eid = dict()
    for edge_type in list(edge_index_dict.keys()):
        edge_index = edge_index_dict[edge_type]
        offset0 = offsets_dict[edge_type[0]]
        offset1 = offsets_dict[edge_type[1]]
        transformed_tensor1 = edge_index[0] - offset0
        transformed_tensor2 = edge_index[1] - offset1
        transformed_eid[edge_type] = torch.tensor([transformed_tensor1.tolist(),
                                                   transformed_tensor2.tolist()])
    return transformed_eid


def metapath2vec_BDT(args):
    """
    prepare data for the metapath2vec Baseline
    :param args:
    :return:
    """
    if args.from_paper == 'GTN':
        name = str(args.dataset).upper()
        root = '/home/ubuntu/msandal_code/PyG_playground/data/IMDB_ACM_DBLP/' + name
        ds = IMDB_ACM_DBLP_from_GTN(root=root,
                                    name=name,
                                    initial_embs=args.acm_dblp_from_gtn_initial_embs,
                                    redownload=args.redownload_data)[0]
    elif args.from_paper == 'NSHE':
        name = str(args.dataset).lower()
        root = '/home/ubuntu/msandal_code/PyG_playground/data/NSHE/' + name
        ds = DBLP_ACM_IMDB_from_NSHE(root=root,
                                     name=name,
                                     redownload=args.redownload_data)[0]
    else:
        raise NotImplementedError('matapath2vec_BDT: unknown paper requested')
    transformed_edge_index = edge_index_dict_individual_ids(ds['edge_index_dict'], ds['node_type_mask'])
    if str(args.dataset).lower() == 'dblp':
        edge_index_dict = dict()
        edge_index_dict[('0_author', 'writes', '1_paper')] = transformed_edge_index[('0', '1')]
        edge_index_dict[('1_paper', 'written_by', '0_author')] = transformed_edge_index[('1', '0')]
        edge_index_dict[('1_paper', 'published_in', '2_conference')] = transformed_edge_index[('1', '2')]
        edge_index_dict[('2_conference', 'publishes', '1_paper')] = transformed_edge_index[('2', '1')]
        metapath = [('0_author', 'writes', '1_paper'),
                    ('1_paper', 'published_in', '2_conference'),
                    ('2_conference', 'publishes', '1_paper'),
                    ('1_paper', 'written_by', '0_author')]
    elif str(args.dataset).lower() == 'acm':
        edge_index_dict = dict()
        edge_index_dict[('0_paper', 'written_by', '1_author')] = transformed_edge_index[('0', '1')]
        edge_index_dict[('1_author', 'writes', '0_paper')] = transformed_edge_index[('1', '0')]
        edge_index_dict[('0_paper', 'features', '2_subject')] = transformed_edge_index[('0', '2')]
        edge_index_dict[('2_subject', 'featured_in', '0_paper')] = transformed_edge_index[('2', '0')]
        metapath = [('1_author', 'writes', '0_paper'),
                    ('0_paper', 'features', '2_subject'),
                    ('2_subject', 'featured_in', '0_paper'),
                    ('0_paper', 'written_by', '1_author')]
    else:
        raise NotImplementedError('metapath2vec_BDT: unknown dataset requested: ' + str(args.dataset))

    return edge_index_dict, ds, metapath


def dataprep_esim(args):
    """
    since ESim (https://github.com/shangjingbo1226/ESim) is a C-based tool,
    here we just generate data structures that are required to create node embeddings
    :param args: experiment arguments
    :return: None
    """
    DATASETS = ['DBLP', 'ACM']
    PAPERS = ['GTN', 'NSHE']

    node_type_int2char = {'ACM': {'0': 'p',
                                  '1': 'a',
                                  '2': 's'},
                          'DBLP': {'0': 'a',
                                   '1': 'p',
                                   '2': 'c'}}
    metapaths = {'DBLP': ['apcpa'], 'ACM': ['apspa']}
    metapath_priorities = {'DBLP': [1], 'ACM': [1]}

    for dataset in DATASETS:
        setattr(args, 'dataset', dataset)
        for paper in PAPERS:
            cprint('Preparing dataset ' + dataset + ' from ' + paper + ' for analysis by ESim',
                   color='cyan')
            setattr(args, 'from_paper', paper)
            ESim_data = '/home/ubuntu/msandal_code/PyG_playground/competitors_perf/baselines/ESim_data'
            ESim_data_folder = os.path.join(ESim_data,
                                            str(args.dataset).lower() + '_' + str(args.from_paper).lower())
            if not os.path.exists(ESim_data_folder):
                Path(ESim_data_folder).mkdir(parents=True, exist_ok=True)

            if args.from_paper == 'GTN':
                name = str(args.dataset).upper()
                root = '/home/ubuntu/msandal_code/PyG_playground/data/IMDB_ACM_DBLP/' + name
                ds = IMDB_ACM_DBLP_from_GTN(root=root,
                                            name=name,
                                            initial_embs=args.acm_dblp_from_gtn_initial_embs,
                                            redownload=args.redownload_data)[0]
            elif args.from_paper == 'NSHE':
                name = str(args.dataset).lower()
                root = '/home/ubuntu/msandal_code/PyG_playground/data/NSHE/' + name
                ds = DBLP_ACM_IMDB_from_NSHE(root=root,
                                             name=name,
                                             redownload=args.redownload_data)[0]
            else:
                raise NotImplementedError('dataprep_esim: unknown paper requested')

            # node file
            node_df = pd.DataFrame({'node_id': list(range(ds['node_type_mask'].shape[0])),
                                    'node_type_int': ds['node_type_mask'].tolist()})
            node_df['node_id'] = node_df['node_id'].apply(lambda x: str(x))
            node_df['node_type_string'] = node_df['node_type_int'].apply(
                lambda x: node_type_int2char[str(args.dataset).upper()][str(x)])
            node_df = node_df[['node_id', 'node_type_string']]
            node_df.to_csv(os.path.join(ESim_data_folder, 'node.dat'),
                           sep=' ',
                           header=False,
                           index=False)

            # link file
            stacked_edges = torch.vstack([ds['edge_index_dict'][key].T for key in list(ds['edge_index_dict'].keys())]).numpy()
            link_df = pd.DataFrame(data=stacked_edges, columns=['source_node', 'target_node'])
            for col in link_df.columns:
                link_df[col] = link_df[col].apply(lambda x: str(x))
            link_df.to_csv(os.path.join(ESim_data_folder, 'link.dat'),
                           sep=' ',
                           header=False,
                           index=False)

            # path file
            path_df = pd.DataFrame({'metapath': metapaths[str(args.dataset).upper()],
                                    'weight': metapath_priorities[str(args.dataset).upper()]})
            path_df['weight'] = path_df['weight'].apply(lambda x: float(x))
            path_df.to_csv(os.path.join(ESim_data_folder, 'path.dat'),
                           sep=' ',
                           header=False,
                           index=False)
