import os
import shutil
import warnings
import torch
import numpy as np
from datetime import datetime

from training_routines import train_rgcn, train_gtn, train_han
from utils.arguments import model_run_argparse
from utils.visualization import draw_embeddings
from torch.utils.tensorboard import SummaryWriter
from downstream_tasks.node_clustering import evaluate_clustering
from downstream_tasks.node_classification import evaluate_classification
from utils.results_recording_local import record_experiment_locally


def run_pipeline(args_, experiment_name_: str = ''):
    print(experiment_name_)
    torch.manual_seed(args_.random_seed)

    # tensorboard: initialize writer; clear the experiment directory if it exists
    if os.path.exists(os.path.join("runs", str(experiment_name_))):
        shutil.rmtree(os.path.join("runs", str(experiment_name_)))
    writer = SummaryWriter(os.path.join("runs", str(experiment_name_)))

    # model training: obtain node embeddings of a given dataset by a give architecture
    # embeddings are stored in :torch.tensor:: output
    if args_.model == 'RGCN':
        output, metadata_ids, metadata_labels, metadata_types, dataset = train_rgcn(args_)
    # elif args_.model == 'GTN':
    #     output, metadata = train_gtn(args_)
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
    id_label_clustering = np.vstack([dataset['train_id_label'],
                                     dataset['test_id_label'],
                                     dataset['valid_id_label']])
    id_label_classification_train = np.vstack([dataset['train_id_label'],
                                               dataset['valid_id_label']])
    id_label_classification_test = dataset['test_id_label']
    # --> clustering, NMI, ARI metrics of K-means
    NMI, ARI = evaluate_clustering(output, ids=id_label_clustering[:, 0],
                                   labels=id_label_clustering[:, 1])
    # --> classification, microF1, macroF1 metrics of logreg
    microF1, macroF1 = evaluate_classification(output,
                                               ids_train=id_label_classification_train[:, 0],
                                               labels_train=id_label_classification_train[:, 1],
                                               ids_test=id_label_classification_test[:, 0],
                                               labels_test=id_label_classification_test[:, 1])

    warnings.simplefilter('default')
    performances_dict = {"NMI": NMI, "ARI": ARI,
                         "microF1": microF1, "macroF1": macroF1}

    # recording experiment results locally
    record_experiment_locally(os.getcwd(), experiment_name_, output, performances_dict, args_,
                              additional_info=str(args_.model) + "_typeAwareBool" + str(args_.type_aware_loss))


if __name__ == "__main__":
    args = model_run_argparse()
    special_notes = 'testing_viz'
    experiment_name = '_'.join([args.dataset, 'from', args.from_paper,
                                args.model,
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
