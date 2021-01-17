import os
import shutil
import pickle as pkl
import re
import torch
import pandas as pd
import scipy as scp
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from scipy.sparse import load_npz
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stopwords

import numpy as np


class DBLP_MAGNN(InMemoryDataset):
    github_url = "https://raw.github.com/cynricfu/MAGNN/master/data/raw/DBLP/"
    feats_url = "https://dl.dropboxusercontent.com/s/yh4grpeks87ugr2/DBLP_processed.zip"

    def __init__(self, root, transform=None, pre_transform=None,
                 use_MAGNN_init_feats: bool = False):
        """
        :param root: see PyG docs
        :param transform: see PyG docs
        :param pre_transform: see PyG docs
        :param use_MAGNN_init_feats: whether to use the initial node embeddings from MAGNN paper
        """
        self.use_MAGNN_init_feats = use_MAGNN_init_feats
        super(DBLP_MAGNN, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['author.txt', 'paper.txt', 'conf.txt', 'term.txt',
                'paper_author.txt', 'paper_conf.txt', 'paper_term.txt',
                'author_label.txt', 'conf_label.txt', 'paper_label.txt',
                'features_0.npz', 'features_1.npz', 'features_2.npy']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        for filename in self.raw_file_names:
            if 'features' in filename:
                continue
            _ = download_url(self.github_url + filename, self.raw_dir)
        if self.use_MAGNN_init_feats:
            path = download_url(self.feats_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)
            for file in os.listdir(self.raw_dir):
                if file not in self.raw_file_names:
                    if os.path.isfile(os.path.join(self.raw_dir, file)):
                        os.unlink(os.path.join(self.raw_dir, file))
                    elif os.path.isdir(os.path.join(self.raw_dir, file)):
                        shutil.rmtree(os.path.join(self.raw_dir, file))

    def process(self):
        DATA_DIR = self.raw_dir

        author_label = pd.read_csv(os.path.join(DATA_DIR, 'author_label.txt'), sep='\t', header=None,
                                   names=['author_id', 'label', 'author_name'], keep_default_na=False, encoding='utf-8')
        conf_label = pd.read_csv(os.path.join(DATA_DIR, 'conf_label.txt'), sep='\t', header=None,
                                 names=['conf_id', 'label'], keep_default_na=False, encoding='utf-8')
        paper_label = pd.read_csv(os.path.join(DATA_DIR, 'paper_label.txt'), sep='\t', header=None,
                                  names=['paper_id', 'label'], keep_default_na=False, encoding='utf-8')
        paper_author = pd.read_csv(os.path.join(DATA_DIR, 'paper_author.txt'), sep='\t', header=None,
                                   names=['paper_id', 'author_id'], keep_default_na=False, encoding='utf-8')
        paper_conf = pd.read_csv(os.path.join(DATA_DIR, 'paper_conf.txt'), sep='\t', header=None,
                                 names=['paper_id', 'conf_id'], keep_default_na=False, encoding='utf-8')
        paper_term = pd.read_csv(os.path.join(DATA_DIR, 'paper_term.txt'), sep='\t', header=None,
                                 names=['paper_id', 'term_id'], keep_default_na=False, encoding='utf-8')
        papers = pd.read_csv(os.path.join(DATA_DIR, 'paper.txt'), sep='\t', header=None,
                             names=['paper_id', 'paper_title'], keep_default_na=False, encoding='cp1252')
        terms = pd.read_csv(os.path.join(DATA_DIR, 'term.txt'), sep='\t', header=None,
                            names=['term_id', 'term'], keep_default_na=False, encoding='utf-8')
        confs = pd.read_csv(os.path.join(DATA_DIR, 'conf.txt'), sep='\t', header=None,
                            names=['conf_id', 'conf'], keep_default_na=False, encoding='utf-8')

        # leave only nodes related to labeled authors
        labeled_authors = author_label['author_id'].to_list()
        paper_author = paper_author[paper_author['author_id'].isin(labeled_authors)].reset_index(drop=True)
        valid_papers = paper_author['paper_id'].unique()
        papers = papers[papers['paper_id'].isin(valid_papers)].reset_index(drop=True)
        paper_label = paper_label[paper_label['paper_id'].isin(valid_papers)].reset_index(drop=True)
        paper_conf = paper_conf[paper_conf['paper_id'].isin(valid_papers)].reset_index(drop=True)
        paper_term = paper_term[paper_term['paper_id'].isin(valid_papers)].reset_index(drop=True)
        valid_terms = paper_term['term_id'].unique()
        terms = terms[terms['term_id'].isin(valid_terms)].reset_index(drop=True)

        terms, paper_term = self.filter_terms(terms, paper_term)

        author_label = author_label.sort_values('author_id', ascending=True).reset_index(drop=True)
        papers = papers.sort_values('paper_id', ascending=True).reset_index(drop=True)
        terms = terms.sort_values('term_id', ascending=True).reset_index(drop=True)
        confs = confs.sort_values('conf_id', ascending=True).reset_index(drop=True)

        init_feats_dict = dict()
        type_file_dict = {"author": "features_0.npz", "paper": "features_1.npz", "term": "features_2.npy"}
        if self.use_MAGNN_init_feats:
            for node_type in type_file_dict.keys():
                if node_type in ['author', 'paper']:
                    init_feats_dict[node_type] = load_npz(
                        os.path.join(self.raw_dir, type_file_dict[node_type])).toarray()
                else:
                    init_feats_dict[node_type] = np.load(os.path.join(self.raw_dir, type_file_dict[node_type]))

        # Preparing dataset components

        # ---> initial node embeddings
        if self.use_MAGNN_init_feats:
            initial_embeddings = dict()
            initial_embeddings['author'] = list(zip(author_label.author_id.tolist(), init_feats_dict['author']))
            initial_embeddings['paper'] = list(zip(papers.paper_id.tolist(), init_feats_dict['paper']))
            initial_embeddings['term'] = list(zip(terms.term_id.tolist(), init_feats_dict['term']))
        else:
            initial_embeddings = None

            # ---> 'verbal' features: paper names, terms, conf names, author names
        node_id_bag_of_words = dict()
        node_id_bag_of_words['author'] = author_label[['author_id', 'author_name']]
        node_id_bag_of_words['paper'] = papers[['paper_id', 'paper_title']]
        node_id_bag_of_words['term'] = terms[['term_id', 'term']]
        node_id_bag_of_words['conf'] = confs[['conf_id', 'conf']]

        # ---> merge new node ids into edges, create corresponding dict as in torch_geometric.datasets.AMiner
        edge_index_dict = dict()
        edge_index_dict[('paper', 'author')] = paper_author[['paper_id', 'author_id']]
        edge_index_dict[('paper', 'term')] = paper_term[['paper_id', 'term_id']]
        edge_index_dict[('paper', 'conf')] = paper_conf[['paper_id', 'conf_id']]

        # ---> for each node type, keep a frame: [[node_new_id], [label]]
        labeled_dict = dict()
        labeled_dict['author'] = author_label[['author_id', 'label']]
        labeled_dict['paper'] = paper_label[['paper_id', 'label']]
        labeled_dict['conf'] = conf_label[['conf_id', 'label']]

        data_list = [Data(node_id_bag_of_words=node_id_bag_of_words,
                          edge_index_dict=edge_index_dict,
                          id_label=labeled_dict,
                          initial_embeddings=initial_embeddings)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @staticmethod
    def filter_terms(terms: pd.DataFrame, paper_term: pd.DataFrame):
        nltk.download('wordnet')
        nltk.download('stopwords')
        # term lemmatization and grouping
        lemmatizer = WordNetLemmatizer()
        lemma_id_mapping = {}
        lemma_list = []
        lemma_id_list = []
        i = 0
        for _, row in terms.iterrows():
            i += 1
            lemma = lemmatizer.lemmatize(row['term'])
            lemma_list.append(lemma)
            if lemma not in lemma_id_mapping:
                lemma_id_mapping[lemma] = row['term_id']
            lemma_id_list.append(lemma_id_mapping[lemma])
        terms['lemma'] = lemma_list
        terms['lemma_id'] = lemma_id_list

        term_lemma_mapping = {row['term_id']: row['lemma_id'] for _, row in terms.iterrows()}
        lemma_id_list = []
        for _, row in paper_term.iterrows():
            lemma_id_list.append(term_lemma_mapping[row['term_id']])
        paper_term['lemma_id'] = lemma_id_list

        paper_term = paper_term[['paper_id', 'lemma_id']]
        paper_term.columns = ['paper_id', 'term_id']
        paper_term = paper_term.drop_duplicates()
        terms = terms[['lemma_id', 'lemma']]
        terms.columns = ['term_id', 'term']
        terms = terms.drop_duplicates()
        stopwords = sklearn_stopwords.union(set(nltk_stopwords.words('english')))
        stopword_id_list = terms[terms['term'].isin(stopwords)]['term_id'].to_list()
        paper_term = paper_term[~(paper_term['term_id'].isin(stopword_id_list))].reset_index(drop=True)
        terms = terms[~(terms['term'].isin(stopwords))].reset_index(drop=True)
        return terms, paper_term


class IMDB_ACM_DBLP(InMemoryDataset):
    ggl_drive_url = 'https://drive.google.com/uc?export=download&id=1qOZ3QjqWMIIvWjzrIdRe3EA4iKzPi6S5'

    def __init__(self, root: str, name: str, transform=None, pre_transform=None):
        """
        :param root: see PyG docs
        :param name: name of the dataset to fetch. must be one of ['DBLP', 'IMDB', 'ACM']
        :param transform: see PyG docs
        :param pre_transform: see PyG docs
        """
        self.ds_name = name if name in ['DBLP', 'IMDB', 'ACM'] else None
        if self.ds_name is None:
            raise ValueError('IMDB_ACM_DBLP.__init__(): name argument invalid!')
        super(IMDB_ACM_DBLP, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['edges.pkl', 'labels.pkl', 'node_features.pkl']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.ggl_drive_url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)
        for folder in os.listdir(self.raw_dir):
            if folder != self.ds_name:
                shutil.rmtree(os.path.join(self.raw_dir, folder))
        for file in os.listdir(os.path.join(self.raw_dir, self.ds_name)):
            shutil.move(os.path.join(self.raw_dir, self.ds_name, file), self.raw_dir)
        shutil.rmtree(os.path.join(self.raw_dir, self.ds_name))

    def process(self):
        data_dict = dict()
        for file in self.raw_file_names:
            with open(os.path.join(self.raw_dir, file), 'rb') as f:
                data_dict[re.sub('.pkl', '', file)] = pkl.load(f)
        node_type_mask = IMDB_ACM_DBLP.infer_type_mask_from_edges(data_dict['edges'])
        data_list = [Data(node_type_mask=node_type_mask,
                          node_features=torch.tensor(data_dict['node_features']))]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

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
                mask = [IMDB_ACM_DBLP.check_interval_overlap(tup, elem) for elem in tups]
                merged.append(IMDB_ACM_DBLP.merge_tups(list(np.array(tups)[mask])))
            merged = list(set(merged))
            for tup in merged:
                mask = [IMDB_ACM_DBLP.check_interval_overlap(tup, elem) for elem in merged]
                true_counts = len([elem for elem in mask if elem == True])
                if true_counts == 1:
                    finished = True
                    continue
                else:
                    finished = False
                    tups = merged
                    merged = list()
                    break
        return sorted(merged, key=lambda tup: tup[0])

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
        tups = IMDB_ACM_DBLP.infer_id_ranges(tups)
        node_type_mask = list()
        for i in range(len(tups)):
            node_type_mask = node_type_mask + [i] * (tups[i][1] - tups[i][0] + 1)
        return torch.tensor(node_type_mask)
