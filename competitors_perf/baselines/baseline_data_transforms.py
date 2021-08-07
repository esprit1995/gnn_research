from datasets import *

def edge_index_dict_individual_ids(edge_index_dict: dict, node_type_mask: torch.tensor):
    """
    make ids of each type start with 0
    :param edge_index_dict: edge_index_dict with keys of type ('int_1', 'int_2')
    :param node_type_mask: node type mask for the given graph
                           node_types are expected to be ascendingly sorted (e.g. [0, ..., 0, 1, ..., 1, 2, ...])
    :return: transformed edge_index_dict
    """
    nnodes_dict = {str(val.item()): len([elem for elem in node_type_mask if elem == val]) for val in node_type_mask.unique()}
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
