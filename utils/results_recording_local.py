import os
import numpy as np
import argparse
import torch
import json
import matplotlib.pyplot as plt

from pathlib import Path
from termcolor import cprint


def record_experiment_locally(project_root_path: str, experiment_name: str,
                              embeddings: torch.tensor, perf_dict: dict,
                              epoch_vs_perf: list,
                              args: argparse.Namespace,
                              additional_info: str = '', ):
    """
    record the results of an experiment (currently, only the node embeddings for a certain dataset);
    with function parameters fixed, the results will be saved to:
    project_root_path/experiment_records/[experiment_name]/[dataset_name]
    If the said experiment on the said dataset had already been run before, the results will be overwritten
    :param project_root_path: root directory of the project
    :param experiment_name: name of the experiment. Each experiment gets a folder
    :param embeddings: node embeddings obtained during the experiment
    :param perf_dict: dictionary containing the performance metrics of downstream tasks on the produced embeddings
    :param epoch_vs_perf: perf tracking over #epochs trained
    :param args: arguments from Argparse used to obtain the result
    :param additional_info: additional information that can help identify the results. For instance, could be model-loss combination
    :return:
    """
    # if the path does not exist, create it
    if not os.path.exists(os.path.join(project_root_path, "experiment_records",  experiment_name)):
        Path(os.path.join(project_root_path, "experiment_records", experiment_name)).mkdir(parents=True, exist_ok=True)

    # if the path exists, empty the directory
    for filename in os.listdir(os.path.join(project_root_path, "experiment_records", experiment_name)):
        os.unlink(os.path.join(project_root_path, "experiment_records", experiment_name, filename))

    # draw epoch vs perf plot
    plt.plot(epoch_vs_perf[0], epoch_vs_perf[1], color='r')
    plt.title(str(args.dataset) + ' from ' + str(args.from_paper) + ', ' + str(args.model))
    plt.xlabel('#epochs trained')
    plt.ylabel('NMI')
    plt.savefig(os.path.join(project_root_path, "experiment_records", experiment_name, 'NMI_vs_n_epochs.png'))

    # save node embeddings
    try:
        np.savez(os.path.join(project_root_path, "experiment_records", experiment_name, "node_embeddings_" + str(additional_info) + ".npz"),
                 embeddings.detach().numpy())
    except Exception as e:
        cprint("Failed to save node embeddings. Exception: " + str(e), color='red', attrs=['bold'])
        return

    # save performance metrics
    try:
        with open(os.path.join(project_root_path, "experiment_records", experiment_name,
                               "perf_metrics_" + str(additional_info) + ".txt"), 'w+') as f:
            json.dump(perf_dict, f, indent=2)
    except Exception as e:
        cprint("Node embeddings saved", color='green')
        cprint("Failed to save performance metrics. Exception: " + str(e), color='red', attrs=['bold'])
        return

    # save argparse arguments
    try:
        with open(os.path.join(project_root_path, "experiment_records", experiment_name, "argparse_args_" + str(additional_info) + ".txt"), 'w+') as f:
            json.dump(args.__dict__, f, indent=2)
    except Exception as e:
        cprint("Node embeddings saved", color='green')
        cprint("Performance metrics saved", color='green')
        cprint("Failed to save argparse arguments. Exception: " + str(e), color='red', attrs=['bold'])
        return
    cprint("Results recorded", color='green')


