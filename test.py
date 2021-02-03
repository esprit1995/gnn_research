import os
import shutil
import warnings
import torch
from training_routines import train_rgcn, train_gtn, train_han
from utils.arguments import model_run_argparse
from torch.utils.tensorboard import SummaryWriter
from downstream_tasks.node_clustering import evaluate_clustering
from downstream_tasks.node_classification import evaluate_classification
from utils.results_recording_local import record_experiment_locally

args = model_run_argparse()
special_notes = ''
experiment_name = '_'.join([args.dataset,
                            args.model,
                            str(args.epochs), 'epochs',
                            str(args.type_aware_loss), 'typeAwareness',
                            str(args.type_lambda), 'lambda',
                            str(args.random_seed), 'rs',
                            special_notes])
torch.manual_seed(args.random_seed)

# tensorboard: initialize writer; clear the experiment directory if it exists
if os.path.exists(os.path.join("runs", str(experiment_name))):
    shutil.rmtree(os.path.join("runs", str(experiment_name)))
writer = SummaryWriter(os.path.join("runs", str(experiment_name)))

# model training: obtain node embeddings of a given dataset by a give architecture
# embeddings are stored in :torch.tensor:: output
if args.model == 'RGCN':
    output, metadata = train_rgcn(args)
elif args.model == 'GTN':
    output, metadata = train_gtn(args)
elif args.model == 'HAN':
    output, metadata = train_han(args)
else:
    raise NotImplementedError("No implementation for model name: ", args.model)

writer.add_embedding(output, metadata=metadata)
writer.flush()
writer.close()


# downstream tasks evaluation
warnings.simplefilter('ignore')
# --> clustering, NMI, ARI metrics of K-means
NMI, ARI = evaluate_clustering(args, output)
# --> classification, microF1, macroF1 metrics of logreg
microF1, macroF1 = evaluate_classification(args, output)
warnings.simplefilter('default')
performances_dict = {"NMI": NMI, "ARI": ARI,
                     "microF1": microF1, "macroF1": macroF1}

# recording experiment results locally
record_experiment_locally(os.getcwd(), experiment_name, args.dataset, output, performances_dict, args,
                          additional_info=str(args.model) + "_typeAwareBool" + str(args.type_aware_loss))
