import os
import shutil
import warnings
import torch
import numpy as np

import tensorflow as tf
import tensorboard as tb

from datetime import datetime

from training_routines import train_rgcn, train_gtn, train_nshe, train_magnn
from utils.arguments import model_run_argparse
from utils.visualization import draw_embeddings
from torch.utils.tensorboard import SummaryWriter
from downstream_tasks.evaluation_funcs import evaluate_clu_cla_GTN_NSHE_datasets
from utils.results_recording_local import record_experiment_locally


def run_pipeline(args_, experiment_name_: str = ''):
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
    print(experiment_name_)
    torch.manual_seed(args_.random_seed)

    # tensorboard: initialize writer; clear the experiment directory if it exists
    if os.path.exists(os.path.join("runs", str(experiment_name_))):
        shutil.rmtree(os.path.join("runs", str(experiment_name_)))
    writer = SummaryWriter(os.path.join("runs", str(experiment_name_)))

    # model training: obtain node embeddings of a given dataset by a give architecture
    # embeddings are stored in :torch.tensor:: output
    if args_.model == 'RGCN':
        output, metadata_ids, metadata_labels, metadata_types, dataset, epochs_num, metrics = train_rgcn(args_)
    elif args_.model == 'GTN':
        output, metadata_ids, metadata_labels, metadata_types, dataset, epochs_num, metrics = train_gtn(args_)
    elif args_.model == 'NSHE':
        output, metadata_ids, metadata_labels, metadata_types, dataset, epochs_num, metrics = train_nshe(args_)
    elif args_.model == 'MAGNN':
        output, metadata_ids, metadata_labels, metadata_types, dataset, epochs_num, metrics = train_magnn(args_)
    # elif args_.model == 'HAN':
    #     output, metadata = train_han(args_)
    else:
        raise NotImplementedError("No implementation for model name: ", args_.model)

    writer.add_embedding(output[metadata_ids], metadata=metadata_labels)
    draw_embeddings(embeddings=output[metadata_ids],
                    cluster_labels=metadata_labels,
                    node_type_mask=metadata_types[metadata_ids],
                    path_to_save=os.path.join('.', 'html_viz'),
                    name_to_save=experiment_name_ + '.html')
    writer.flush()
    writer.close()

    # downstream tasks evaluation
    warnings.simplefilter('ignore')
    NMI, ARI, microF1, macroF1 = evaluate_clu_cla_GTN_NSHE_datasets(dataset, output)
    best_NMI = max(metrics)
    best_NMI_at_epoch = epochs_num[metrics.index(max(metrics))]
    print('best performance (NMI clu) achieved at epoch: ' + str(best_NMI_at_epoch))
    print('and is equal to: ' + str(best_NMI))
    warnings.simplefilter('default')
    performances_dict = {"NMI": NMI, "ARI": ARI,
                         "microF1": microF1, "macroF1": macroF1,
                         "best_NMI": best_NMI, "best_NMI_at_epoch": best_NMI_at_epoch}

    # recording experiment results locally
    record_experiment_locally(os.getcwd(), experiment_name_, output, performances_dict,
                              [epochs_num, metrics], args_,
                              additional_info=str(args_.model) + "_typeAwareBool" + str(args_.type_aware_loss))


if __name__ == "__main__":
    args = model_run_argparse()
    special_notes = ''
    if args.dataset in ['DBLP', 'ACM'] and args.from_paper == 'GTN':
        dataset_name = '_'.join([args.dataset, 'from', args.from_paper, 'initial_embs', args.acm_dblp_from_gtn_initial_embs])
    else:
        dataset_name = '_'.join([args.dataset, 'from', args.from_paper])
    experiment_name = '_'.join([dataset_name,
                                'Model', args.model,
                                str(args.epochs), 'epochs',
                                str(args.base_triplet_loss), 'baseTripletLoss',
                                str(args.cocluster_loss), 'coclusterLoss',
                                str(args.type_lambda), 'lambda',
                                str(args.corruption_method), 'corrmethod',
                                str(args.instances_per_template), 'ipt',
                                str(args.random_seed), 'rs',
                                datetime.now().strftime("%d_%m_%Y_%H_%M_%S"),
                                special_notes])
    run_pipeline(args, experiment_name)
