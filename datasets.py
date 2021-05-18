import os
import shutil
import pickle as pkl
import re
import requests

import torch
import dgl
import pandas as pd
import numpy as np
import scipy as scp

from termcolor import cprint
from pathlib import Path

from scipy import io as sio
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split


class IMDB_ACM_DBLP_from_GTN(InMemoryDataset):
    # '1' = paper, '0' = author, '2' = conference for DBLP
    # '0' = paper, '1' = 'author', '2' = subject for ACM
    ggl_drive_url = 'https://drive.google.com/uc?export=download&id=1qOZ3QjqWMIIvWjzrIdRe3EA4iKzPi6S5'
    dblp_additional = 'https://raw.github.com/cynricfu/MAGNN/master/data/raw/DBLP/'

    deepwalk_embs_link = 'https://raw.github.com/esprit1995/useful_files_repo/master/msc_thesis_files/'

    def __init__(self, root: str, name: str, initial_embs: str='original',
                 multi_type_labels: bool = True, redownload: bool = False,
                 transform=None, pre_transform=None):
        """
        :param root: see PyG docs
        :param name: name of the dataset to fetch. must be one of ['DBLP', 'IMDB', 'ACM']
        :param initial_embs: which initial node embeddings to use. must be one of ['original', 'deepwalk']
        :param multi_type_labels: whether to infer additional labels for multi-type clustering tasks (for DBLP, ACM)
        :param redownload: whether to redownload/reprocess the data from scratch
        :param transform: see PyG docs
        :param pre_transform: see PyG docs
        """
        if redownload and os.path.exists(root):
            shutil.rmtree(root)
        # delete processed data to force processing anew:
        # needed since there are some args that affect the output
        if os.path.exists(os.path.join(root, 'processed')):
            shutil.rmtree(os.path.join(root, 'processed'))
        self.initial_embs_name = initial_embs
        self.multi_type_labels = multi_type_labels
        self.ds_name = name if name in ['DBLP', 'IMDB', 'ACM'] else None
        if self.ds_name is None:
            raise ValueError('IMDB_ACM_DBLP.__init__(): name argument invalid!')
        super(IMDB_ACM_DBLP_from_GTN, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        if self.ds_name == 'ACM':
            return ['edges.pkl', 'labels.pkl', 'node_features.pkl',
                    'acm_from_gtn_deepwalk.embeddings']
        elif self.ds_name == 'IMDB':
            return ['edges.pkl', 'labels.pkl', 'node_features.pkl']
        else:
            return ['edges.pkl', 'labels.pkl', 'node_features.pkl',
                    'author_label.txt', 'conf_label.txt',
                    'paper_conf.txt', 'paper_author.txt',
                    'dblp_from_gtn_deepwalk.embeddings']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.ggl_drive_url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

        deepwalk_embs_filename = self.ds_name.lower() + '_from_gtn_deepwalk.embeddings'
        _ = download_url(os.path.join(self.deepwalk_embs_link, deepwalk_embs_filename),
                         self.raw_dir)
        shutil.move(os.path.join(self.raw_dir, deepwalk_embs_filename),
                    os.path.join(self.raw_dir, self.ds_name))
        for folder in os.listdir(self.raw_dir):
            if folder != self.ds_name:
                shutil.rmtree(os.path.join(self.raw_dir, folder))
        for file in os.listdir(os.path.join(self.raw_dir, self.ds_name)):
            shutil.move(os.path.join(self.raw_dir, self.ds_name, file), self.raw_dir)
        shutil.rmtree(os.path.join(self.raw_dir, self.ds_name))
        if self.ds_name == 'DBLP':
            for file in self.raw_file_names:
                if '.pkl' not in file and '_deepwalk.embeddings' not in file:
                    _ = download_url(self.dblp_additional + file, self.raw_dir)

    def process(self):
        data_dict = dict()
        if self.initial_embs_name == 'deepwalk':
            if self.ds_name == 'IMDB':
                raise ValueError('IMDB does not have DeepWalk embeddings option')
            emb_filename = self.ds_name.lower() + "_from_gtn_deepwalk.embeddings"
            with open(os.path.join(self.raw_dir, emb_filename)) as f:
                lines = (line for line in f)
                embs = np.loadtxt(lines, delimiter=' ', skiprows=1)
            embs = embs[np.argsort(embs[:, 0])]  # sort by node id, which is the first column
            embs = embs[:, 1:]  # remove the first column, as it is not a feature
            data_dict['node_features'] = embs
            for file in self.raw_file_names:
                if '.pkl' not in file or file == 'node_features.pkl':
                    continue
                with open(os.path.join(self.raw_dir, file), 'rb') as f:
                    data_dict[re.sub('.pkl', '', file)] = pkl.load(f)
        elif self.initial_embs_name == 'original':
            for file in self.raw_file_names:
                if '.pkl' not in file:
                    continue
                with open(os.path.join(self.raw_dir, file), 'rb') as f:
                    data_dict[re.sub('.pkl', '', file)] = pkl.load(f)
        else:
            raise ValueError('initial_embs argument faulty: must be on of ["original", "deepwalk"]')
        node_type_mask = IMDB_ACM_DBLP_from_GTN.infer_type_mask_from_edges(data_dict['edges'])
        edge_index_dict = IMDB_ACM_DBLP_from_GTN.get_edge_index_dict(data_dict['edges'], node_type_mask)
        train_id_label = np.array(data_dict['labels'][0])
        valid_id_label = np.array(data_dict['labels'][1])
        test_id_label = np.array(data_dict['labels'][2])

        # infer additional labels for multi-type clustering tasks
        if self.multi_type_labels:
            if self.ds_name == 'ACM':
                paper_author = pd.DataFrame(columns=['paper', 'author'],
                                            data=edge_index_dict[('0', '1')].numpy().T)
                paper_label = pd.DataFrame(columns=['paper', 'label'],
                                           data=np.vstack([train_id_label,
                                                           valid_id_label,
                                                           test_id_label]))
                author_label = paper_author.merge(paper_label, on='paper', how='inner').drop(columns=['paper'])
                author_label = author_label.groupby(['author']) \
                    .agg({'label': lambda labels: labels.value_counts().index[0]}) \
                    .reset_index(drop=False)
                node_id_node_label = np.vstack([author_label.to_numpy(),
                                                train_id_label,
                                                valid_id_label,
                                                test_id_label])
            if self.ds_name == 'DBLP':
                paper_author = pd.DataFrame(columns=['paper', 'author'],
                                            data=IMDB_ACM_DBLP_from_GTN.read_2colInt_txt(
                                                open(os.path.join(self.raw_dir, 'paper_author.txt'), 'r')))
                paper_conf = pd.DataFrame(columns=['paper', 'conf'],
                                          data=IMDB_ACM_DBLP_from_GTN.read_2colInt_txt(
                                              open(os.path.join(self.raw_dir, 'paper_conf.txt'), 'r')))
                author_label = pd.DataFrame(columns=['author', 'label'],
                                            data=IMDB_ACM_DBLP_from_GTN.read_2colInt_txt(
                                                open(os.path.join(self.raw_dir, 'author_label.txt'), 'r')))
                conf_label = pd.DataFrame(columns=['conf', 'label'],
                                          data=IMDB_ACM_DBLP_from_GTN.read_2colInt_txt(
                                              open(os.path.join(self.raw_dir, 'conf_label.txt'), 'r')))

                paper_author = paper_author[paper_author['author'].isin(author_label['author'].tolist())]
                paper_conf = paper_conf[paper_conf['paper'].isin(paper_author['paper'].tolist())]
                paper_conf = paper_conf.groupby(['conf'])['paper'].count().to_frame().reset_index()
                paper_conf_local = pd.DataFrame(columns=['paper', 'conf_local'],
                                                data=edge_index_dict[('1', '2')].numpy().T)
                paper_conf_local = paper_conf_local.groupby(['conf_local'])['paper'].count().to_frame().reset_index()
                conf_local_conf = paper_conf.merge(paper_conf_local, on='paper', how='inner')[['conf_local', 'conf']]
                conf_local_label = conf_local_conf.merge(conf_label, on='conf', how='inner')[['conf_local', 'label']]

                local_conf_paper = pd.DataFrame(columns=['paper', 'conf_local'],
                                                data=edge_index_dict[('1', '2')].numpy().T)
                paper_label = local_conf_paper.merge(conf_local_label, on='conf_local', how='inner')[['paper', 'label']]
                node_id_node_label = np.vstack([paper_label.to_numpy(),
                                                train_id_label,
                                                valid_id_label,
                                                test_id_label])

            train_id_label, test_id_label = train_test_split(node_id_node_label,
                                                             test_size=0.7,
                                                             shuffle=True,
                                                             stratify=node_id_node_label[:, 1],
                                                             random_state=0)
            train_id_label, valid_id_label = train_test_split(train_id_label,
                                                              test_size=0.3,
                                                              shuffle=True,
                                                              stratify=train_id_label[:, 1],
                                                              random_state=0)

        data_list = [Data(node_type_mask=node_type_mask,
                          node_features=torch.tensor(data_dict['node_features']),
                          edge_index_dict=edge_index_dict,
                          train_id_label=train_id_label,
                          valid_id_label=valid_id_label,
                          test_id_label=test_id_label)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @staticmethod
    def get_edge_index_dict(edges: list, node_type_mask):
        edge_index_dict = dict()
        for edge_type_idx in range(len(edges)):
            non_zero_indices = np.nonzero(np.asarray(edges[edge_type_idx].todense()))
            ntype1 = node_type_mask[non_zero_indices[0][0]].item()
            ntype2 = node_type_mask[non_zero_indices[1][0]].item()
            edge_index_dict[(str(ntype1), str(ntype2))] = torch.tensor(non_zero_indices)
        return edge_index_dict

    @staticmethod
    def read_2colInt_txt(file):
        res = list()
        for line in file:
            elems = str(line).split('\t')
            val1, val2 = int(elems[0]), int(elems[1])
            res.append([val1, val2])
        return np.array(res)

    @staticmethod
    def check_interval_overlap(int1, int2):
        cond1 = (int2[0] <= int1[0] <= int2[1])
        cond2 = (int2[0] <= int1[1] <= int2[1])
        cond3 = (int1[0] <= int2[0] and int1[1] >= int2[1])
        return cond1 or cond2 or cond3

    @staticmethod
    def merge_tups(tups):
        return min([x[0] for x in tups]), max([x[1] for x in tups])

    @staticmethod
    def infer_id_ranges(tups):
        merged = list()
        finished = False
        while not finished:
            for tup in tups:
                mask = [IMDB_ACM_DBLP_from_GTN.check_interval_overlap(tup, elem) for elem in tups]
                merged.append(IMDB_ACM_DBLP_from_GTN.merge_tups(list(np.array(tups)[mask])))
            merged = list(set(merged))
            for tup in merged:
                mask = [IMDB_ACM_DBLP_from_GTN.check_interval_overlap(tup, elem) for elem in merged]
                true_counts = len([elem for elem in mask if elem == True])
                if true_counts == 1:
                    finished = True
                    continue
                else:
                    finished = False
                    tups = merged
                    merged = list()
                    break
        return sorted(merged, key=lambda tup_: tup_[0])

    @staticmethod
    def infer_type_mask_from_edges(edges: scp.sparse.csr.csr_matrix):
        """
        the raw data does not contain node type mask. However, it can be inferred from the edges matrix
        some assumptions (verified):
        1. shape(edges) = n_edge_types x n_nodes x n_nodes
        2. each edges[i] stores adjacency matrix for a fixed type of edge
        3. node ids of the same type nodes go uninterrupted (for instance, expected node type mask
           resembles [0, 0, ..., 0, 1, ..., 1, 2, ..., 2, ...]
        :param edges: edges[i] contains a ndarray of edges of fixed type.
        :return:
        """
        num_nodes = edges[1].todense().shape[0]
        edge_type_ids_dict = dict()
        for edge_type in range(len(edges)):
            non_zero_idx1 = list()
            non_zero_idx2 = list()
            matrix = np.asarray(edges[edge_type].todense())
            for i in range(num_nodes):
                local_non0 = np.nonzero(matrix[i])[0]
                if local_non0.size > 0:
                    non_zero_idx2.append(i)
                non_zero_idx1 = non_zero_idx1 + local_non0.tolist()
            ids1 = sorted(list(set(non_zero_idx1)))
            ids2 = sorted(non_zero_idx2)
            edge_type_ids_dict[edge_type] = ((min(ids1), max(ids1), len(ids1)),
                                             (min(ids2), max(ids2), len(ids2)))
        tups = list()
        for key in edge_type_ids_dict.keys():
            tups.append(edge_type_ids_dict[key][0])
            tups.append(edge_type_ids_dict[key][1])
        tups = IMDB_ACM_DBLP_from_GTN.infer_id_ranges(tups)
        node_type_mask = list()
        for i in range(len(tups)):
            node_type_mask = node_type_mask + [i] * (tups[i][1] - tups[i][0] + 1)
        return torch.tensor(node_type_mask)


class DBLP_ACM_IMDB_from_NSHE(InMemoryDataset):
    def __init__(self, root, name, redownload: bool = False, transform=None, pre_transform=None):
        """
        :param root: see PyG docs
        :param name: which dataset to fetch. must be one of ['acm', 'dblp', 'imdb']
        :param redownload: whether do redownload data even if it has been processed before
        :param transform: see PyG docs
        :param pre_transform: see PyG docs
        """
        assert name in ['acm', 'dblp', 'imdb'], \
            "DBLP_ACM_IMDB_from_NSHE: name argument must be one of ['acm', 'dblp', 'imdb']"
        if redownload and os.path.exists(root):
            shutil.rmtree(root)
        self.github_url = "https://raw.github.com/Andy-Border/NSHE/master/data"
        self.ds_name = name
        self.data_url = '/'.join([self.github_url, self.ds_name])
        root = os.path.join(root, name)
        if not os.path.exists(root):
            Path(root).mkdir(parents=True, exist_ok=True)
        super(DBLP_ACM_IMDB_from_NSHE, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        if self.ds_name == 'dblp':
            return ['author_label.txt', 'dw_emb_features.npy', 'node2id.txt', 'paper_label.txt',
                    'relation2id.txt', 'relations.txt']
        elif self.ds_name == 'acm':
            return ['dw_emb_features.npy', 'node2id.txt', 'p_label.txt',
                    'relation2id.txt', 'relations.txt']
        elif self.ds_name == 'imdb':
            return ['dw_emb_features.npy', 'node2id.txt', 'imdb_m_label.txt',
                    'relation2id.txt', 'relations.txt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        for filename in self.raw_file_names:
            _ = download_url(self.data_url + '/' + filename, self.raw_dir)

    def process(self):
        node_features = torch.tensor(np.load(os.path.join(self.raw_dir, 'dw_emb_features.npy')))
        if self.ds_name == 'dblp':
            # === node type mask construction, 0=author, 1=paper, 2=conference
            node_type_mask = torch.tensor(np.array([0] * 2000 + [1] * 9556 + [2] * 20))

            # --- edge index dictionary construction
            relations_df = pd.read_csv(os.path.join(self.raw_dir, 'relations.txt'),
                                       sep='\t',
                                       header=None)
            relations_df.columns = ['id1', 'id2', 'edge_type', 'garbage1', 'garbage2']
            relations_df = relations_df[['id1', 'id2', 'edge_type']]
            relations_ap = relations_df[relations_df['edge_type'] == 0]
            relations_pc = relations_df[relations_df['edge_type'] == 1]

            edge_index_dict = dict()
            edge_index_dict[('0', '1')] = torch.tensor(np.array([relations_ap['id1'].to_numpy(),
                                                                 relations_ap['id2'].to_numpy()]))
            edge_index_dict[('1', '0')] = torch.tensor(np.array([relations_ap['id2'].to_numpy(),
                                                                 relations_ap['id1'].to_numpy()]))
            edge_index_dict[('1', '2')] = torch.tensor(np.array([relations_pc['id1'].to_numpy(),
                                                                 relations_pc['id2'].to_numpy()]))
            edge_index_dict[('2', '1')] = torch.tensor(np.array([relations_pc['id2'].to_numpy(),
                                                                 relations_pc['id1'].to_numpy()]))

            # --- labeled nodes procurement
            paper_label = pd.read_csv(os.path.join(self.raw_dir, 'paper_label.txt'),
                                      sep='\t',
                                      header=None,
                                      names=['paper', 'label'])
            paper_label['paper'] = paper_label['paper'].apply(lambda x: 'p' + str(x))
            author_label = pd.read_csv(os.path.join(self.raw_dir, 'author_label.txt'),
                                       sep='\t',
                                       header=None,
                                       names=['author', 'label'])
            author_label['author'] = author_label['author'].apply(lambda x: 'a' + str(x))
            node2id = pd.read_csv(os.path.join(self.raw_dir, 'node2id.txt'),
                                  sep='\t').reset_index()
            node2id.columns = ['node', 'node_id']

            paper_label = paper_label.merge(node2id, how='inner', left_on='paper', right_on='node')
            author_label = author_label.merge(node2id, how='inner', left_on='author', right_on='node')
            node_id_node_label = torch.tensor([paper_label['node_id'].tolist() + author_label['node_id'].tolist(),
                                               paper_label['label'].tolist() + author_label['label'].tolist()])
        elif self.ds_name == 'acm':
            # === node type mask construction, 0=paper, 1=author, 2=subject
            node_type_mask = torch.tensor(np.array([0] * 4019 + [1] * 7167 + [2] * 60))

            # --- edge index dictionary construction
            relations_df = pd.read_csv(os.path.join(self.raw_dir, 'relations.txt'),
                                       sep='\t',
                                       header=None)
            relations_df.columns = ['id1', 'id2', 'edge_type', 'garbage1']
            relations_df = relations_df[['id1', 'id2', 'edge_type']]
            relations_pa = relations_df[relations_df['edge_type'] == 0]
            relations_ps = relations_df[relations_df['edge_type'] == 1]

            edge_index_dict = dict()
            edge_index_dict[('0', '1')] = torch.tensor(np.array([relations_pa['id1'].to_numpy(),
                                                                 relations_pa['id2'].to_numpy()]))
            edge_index_dict[('1', '0')] = torch.tensor(np.array([relations_pa['id2'].to_numpy(),
                                                                 relations_pa['id1'].to_numpy()]))
            edge_index_dict[('0', '2')] = torch.tensor(np.array([relations_ps['id1'].to_numpy(),
                                                                 relations_ps['id2'].to_numpy()]))
            edge_index_dict[('2', '0')] = torch.tensor(np.array([relations_ps['id2'].to_numpy(),
                                                                 relations_ps['id1'].to_numpy()]))
            # --- labeled nodes procurement
            paper_label = pd.read_csv(os.path.join(self.raw_dir, 'p_label.txt'),
                                      sep='\t',
                                      header=None)
            paper_label.columns = ['paper_id', 'label']
            node_id_node_label = torch.tensor([paper_label['paper_id'].tolist(),
                                               paper_label['label'].tolist()])
        elif self.ds_name == 'imdb':
            # === node type mask construction, 0=movie, 1=actor, 2=director
            node_type_mask = torch.tensor(np.array([0] * 3676 + [1] * 4353 + [2] * 1678))

            # --- edge index dictionary construction
            relations_df = pd.read_csv(os.path.join(self.raw_dir, 'relations.txt'),
                                       sep='\t',
                                       header=None)
            relations_df.columns = ['id1', 'id2', 'edge_type', 'garbage1']
            relations_df = relations_df[['id1', 'id2', 'edge_type']]
            relations_ma = relations_df[relations_df['edge_type'] == 0]
            relations_md = relations_df[relations_df['edge_type'] == 1]

            edge_index_dict = dict()
            edge_index_dict[('0', '1')] = torch.tensor(np.array([relations_ma['id1'].to_numpy(),
                                                                 relations_ma['id2'].to_numpy()]))
            edge_index_dict[('1', '0')] = torch.tensor(np.array([relations_ma['id2'].to_numpy(),
                                                                 relations_ma['id1'].to_numpy()]))
            edge_index_dict[('0', '2')] = torch.tensor(np.array([relations_md['id1'].to_numpy(),
                                                                 relations_md['id2'].to_numpy()]))
            edge_index_dict[('2', '0')] = torch.tensor(np.array([relations_md['id2'].to_numpy(),
                                                                 relations_md['id1'].to_numpy()]))

            # --- labeled nodes procurement
            movie_label = pd.read_csv(os.path.join(self.raw_dir, 'imdb_m_label.txt'),
                                      sep='\t',
                                      header=None).reset_index(drop=False)
            movie_label.columns = ['movie_id', 'label']
            node_id_node_label = torch.tensor([movie_label['movie_id'].tolist(),
                                               movie_label['label'].tolist()])
        else:
            raise ValueError('DBLP_ACM_from_NSHE: unknown dataset name')

        # node_labels_dict
        id_label_df = pd.DataFrame(data=node_id_node_label.numpy().T,
                                   columns=['id', 'label'])
        train_id_label, test_id_label = train_test_split(id_label_df.to_numpy(),
                                                         test_size=0.7,
                                                         shuffle=True,
                                                         stratify=id_label_df.to_numpy()[:, 1],
                                                         random_state=0)
        train_id_label, valid_id_label = train_test_split(train_id_label,
                                                          test_size=0.3,
                                                          shuffle=True,
                                                          stratify=train_id_label[:, 1],
                                                          random_state=0)

        data_list = [Data(node_features=node_features,
                          node_type_mask=node_type_mask,
                          edge_index_dict=edge_index_dict,
                          node_id_node_label=node_id_node_label,
                          train_id_label=train_id_label,
                          valid_id_label=valid_id_label,
                          test_id_label=test_id_label)]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class ACM_HAN(InMemoryDataset):
    """
    procure ACM dataset from ACM.mat raw matrix as it is in the
    Heterogeneous Graph Attention Network (HAN) paper
    """

    def __init__(self, root, transform=None, pre_transform=None):
        """
        :param root: see PyG docs
        :param transform: see PyG docs
        :param pre_transform: see PyG docs
        """
        self.github_url = "https://raw.github.com/Jhy1993/HAN/master/data/acm/"
        super(ACM_HAN, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['ACM.mat']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        for filename in self.raw_file_names:
            _ = download_url(self.github_url + filename, self.raw_dir)

    def process(self):
        dgl_hetgraph, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
        val_mask, test_mask = ACM_HAN.load_raw_acm(os.path.join(self.raw_dir, 'ACM.mat'))
        train_id_label = torch.tensor([np.nonzero(train_mask.numpy())[0],
                                       labels[train_mask.type(torch.BoolTensor)].numpy()])
        test_id_label = torch.tensor([np.nonzero(test_mask.numpy())[0],
                                      labels[test_mask.type(torch.BoolTensor)].numpy()])
        data_list = [Data(dgl_hetgraph=dgl_hetgraph,
                          features=features,
                          labels=labels,
                          num_classes=num_classes,
                          train_id_label=train_id_label,
                          test_id_label=test_id_label)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @staticmethod
    def load_raw_acm(data_path: str):
        """
        credits to DGL's official github page examples:
        https://github.com/dmlc/dgl/blob/master/examples/pytorch/han/utils.py
        :param data_path: string indicating the path to ACM.mat file
        :return:
        """
        data = sio.loadmat(data_path)
        p_vs_l = data['PvsL']  # paper-field?
        p_vs_a = data['PvsA']  # paper-author
        p_vs_t = data['PvsT']  # paper-term, bag of words
        p_vs_c = data['PvsC']  # paper-conference, labels come from that

        # We assign
        # (1) KDD papers as class 0 (data mining),
        # (2) SIGMOD and VLDB papers as class 1 (database),
        # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
        conf_ids = [0, 1, 9, 10, 13]
        label_ids = [0, 1, 2, 2, 1]

        p_vs_c_filter = p_vs_c[:, conf_ids]
        p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
        p_vs_l = p_vs_l[p_selected]
        p_vs_a = p_vs_a[p_selected]
        p_vs_t = p_vs_t[p_selected]
        p_vs_c = p_vs_c[p_selected]

        hg = dgl.heterograph({
            ('paper', 'pa', 'author'): p_vs_a.nonzero(),
            ('author', 'ap', 'paper'): p_vs_a.transpose().nonzero(),
            ('paper', 'pf', 'field'): p_vs_l.nonzero(),
            ('field', 'fp', 'paper'): p_vs_l.transpose().nonzero()
        })

        features = torch.FloatTensor(p_vs_t.toarray())

        pc_p, pc_c = p_vs_c.nonzero()
        labels = np.zeros(len(p_selected), dtype=np.int64)
        for conf_id, label_id in zip(conf_ids, label_ids):
            labels[pc_p[pc_c == conf_id]] = label_id
        labels = torch.LongTensor(labels)

        num_classes = 3

        float_mask = np.zeros(len(pc_p))
        for conf_id in conf_ids:
            pc_c_mask = (pc_c == conf_id)
            float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))
        train_idx = np.where(float_mask <= 0.2)[0]
        val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
        test_idx = np.where(float_mask > 0.3)[0]

        num_nodes = hg.number_of_nodes('paper')
        train_mask = ACM_HAN.get_binary_mask(num_nodes, train_idx)
        val_mask = ACM_HAN.get_binary_mask(num_nodes, val_idx)
        test_mask = ACM_HAN.get_binary_mask(num_nodes, test_idx)

        return hg, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask

    @staticmethod
    def get_binary_mask(total_size, indices):
        mask = torch.zeros(total_size)
        mask[indices] = 1
        return mask.byte()


class HNE_DATASETS(InMemoryDataset):
    """
    provides access to the datasets described in the survey article
    "Heterogeneous Network Representation Learning: A Unified Framework
    With Survey And Benchmark"
    Datasets: PubMed, Yelp, DBLP, Freebase
    """
    yelp_id = "1pS78jCtnAnlAQfcfJMkWnJ2utInDu91k"
    dblp_id = "1nrGb7eEQj7h5YC5kv65hcrtIhr7l03AI"
    freebase_id = "15gdrbv9l7luEFHq3YdAkeWYldHcqEjG9"
    pubmed_id = "1ZEi2sTaZ2bk8cQwyCxtlwuWJsAq9N-Cl"

    def __init__(self, root, name, transform=None, pre_transform=None):
        """
        :param root: see PyG docs
        :param name: name of the dataset to procure. Must be one of: ["dblp", "yelp", "freebase", "pubmed"]
        :param transform: see PyG docs
        :param pre_transform: see PyG docs
        """
        if not os.path.exists(root):
            Path(root).mkdir(parents=True, exist_ok=True)
        self.name = name
        super(HNE_DATASETS, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['node.dat', 'link.dat', 'label.dat',
                'link.dat.test', 'label.dat.test',
                'meta.dat', 'info.dat',
                'record.dat']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        cprint('Downloading zip file from ggdrive...', color='blue', attrs=['bold'])
        if str(self.name).lower() == 'yelp':
            HNE_DATASETS.download_file_from_google_drive(self.yelp_id, os.path.join(self.raw_dir, 'yelp.zip'))
            zip_path = os.path.join(self.raw_dir, 'yelp.zip')
        elif str(self.name).lower() == 'dblp':
            HNE_DATASETS.download_file_from_google_drive(self.dblp_id, os.path.join(self.raw_dir, 'dblp.zip'))
            zip_path = os.path.join(self.raw_dir, 'dblp.zip')
        elif str(self.name).lower() == 'freebase':
            HNE_DATASETS.download_file_from_google_drive(self.freebase_id, os.path.join(self.raw_dir, 'freebase.zip'))
            zip_path = os.path.join(self.raw_dir, 'freebase.zip')
        elif str(self.name).lower() == 'pubmed':
            HNE_DATASETS.download_file_from_google_drive(self.pubmed_id, os.path.join(self.raw_dir, 'pubmed.zip'))
            zip_path = os.path.join(self.raw_dir, 'pubmed.zip')
        else:
            raise NotImplementedError("Requested dataset not available:", self.name)

        extract_zip(zip_path, self.raw_dir)
        os.unlink(zip_path)
        datadir = os.path.join(self.raw_dir, os.listdir(self.raw_dir)[0])
        for file in os.listdir(datadir):
            shutil.move(os.path.join(datadir, file), self.raw_dir)
        shutil.rmtree(datadir)

    def process(self):
        attributed = ['dblp', 'pubmed']

        # retrieve node information
        node_ids = list()
        node_types = list()
        node_feats = list()
        file = open(os.path.join(self.raw_dir, 'node.dat'), 'r')
        for line in file:
            line = line[:-1].split('\t')
            node_ids.append(int(line[0]))
            node_types.append(int(line[2]))
            if self.name in attributed:
                node_feats.append([float(elem) for elem in line[3].split(',')])
            else:
                pass

        # retrieve link information
        link_id_source = list()
        link_id_target = list()
        link_types = list()
        link_weights = list()
        file = open(os.path.join(self.raw_dir, 'link.dat'), 'r')
        for line in file:
            line = line[:-1].split('\t')
            link_id_source.append(int(line[0]))
            link_id_target.append(int(line[1]))
            link_types.append(int(line[2]))
            link_weights.append(float(line[3]))
        edge_index = [link_id_source, link_id_target]

        # retrieve label information
        labeling_node_id = list()
        labeling_node_label = list()
        file = open(os.path.join(self.raw_dir, 'label.dat'), 'r')
        for line in file:
            line = line[:-1].split('\t')
            labeling_node_id.append(int(line[0]))
            labeling_node_label.append(int(line[3]))
        labeling = [labeling_node_id, labeling_node_label]

        # retrieve label testing info
        test_label_ids = list()
        test_label_labels = list()
        file = open(os.path.join(self.raw_dir, 'label.dat.test'), 'r')
        for line in file:
            line = line[:-1].split('\t')
            test_label_ids.append(int(line[0]))
            test_label_labels.append(int(line[3]))
        labeling_test = [test_label_ids, test_label_labels]

        # retrieve link testing info
        linked_nodes_test = list()
        unlinked_nodes_test = list()
        file = open(os.path.join(self.raw_dir, 'link.dat.test'), 'r')
        for line in file:
            line = line[:-1].split('\t')
            if int(line[2]) == 1:
                linked_nodes_test.append((line[0], line[1]))
            elif int(line[2]) == 0:
                unlinked_nodes_test.append((line[0], line[1]))
            else:
                cprint('HNE_DATASETS link test parsing warning: unexpected item encountered: ' + str(line[2]),
                       color='yellow')
        linked_nodes_test = [[int(elem[0]) for elem in linked_nodes_test],
                             [int(elem[1]) for elem in linked_nodes_test]]
        unlinked_nodes_test = [[int(elem[0]) for elem in unlinked_nodes_test],
                               [int(elem[1]) for elem in unlinked_nodes_test]]

        data_list = [Data(node_ids=torch.LongTensor(node_ids),
                          node_types=torch.IntTensor(node_types),
                          node_features=torch.FloatTensor([np.array(elem) for elem in node_feats]),
                          edge_index=torch.LongTensor([np.array(elem) for elem in edge_index]),
                          edge_types=torch.IntTensor(link_types),
                          edge_weights=torch.FloatTensor(link_weights),
                          node_id_to_label=torch.LongTensor(labeling),
                          node_id_to_label_test=torch.LongTensor([np.array(elem) for elem in labeling_test]),
                          link_test_id_pairs_positive=torch.LongTensor(linked_nodes_test),
                          link_test_id_pairs_negative=torch.LongTensor(unlinked_nodes_test))]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @staticmethod
    def download_file_from_google_drive(id_, destination):
        def get_confirm_token(response_):
            for key, value in response_.cookies.items():
                if key.startswith('download_warning'):
                    return value

            return None

        def save_response_content(response__, destination__):
            CHUNK_SIZE = 32768

            with open(destination__, "wb") as f:
                for chunk in response__.iter_content(CHUNK_SIZE):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)

        URL = "https://docs.google.com/uc?export=download"

        session = requests.Session()

        response = session.get(URL, params={'id': id_}, stream=True)
        token = get_confirm_token(response)

        if token:
            params = {'id': id_, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        save_response_content(response, destination)
