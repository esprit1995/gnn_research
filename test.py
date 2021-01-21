import os
import warnings
from training_routines import train_rgcn, train_gtn
from utils.arguments import model_run_argparse
from torch.utils.tensorboard import SummaryWriter
from downstream_tasks.node_clustering import evaluate_clustering
from downstream_tasks.node_classification import evaluate_classification
from utils.results_recording_local import  record_experiment_locally

experiment_name = 'testing_results_recording'
args = model_run_argparse()
writer = SummaryWriter("runs/" + str(experiment_name))

# model training: obtain node embeddings of a given dataset by a give architecture
# embeddings are stored in :torch.tensor:: output
if args.model == 'RGCN':
    output, metadata = train_rgcn(args)
elif args.model == 'eGTN':
    output, metadata = train_gtn(args)
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

# recording experiment results locally
record_experiment_locally(os.getcwd(), experiment_name, args.dataset, output, args,
                          additional_info=str(args.model) + "_typeAwareBool" + str(args.type_aware_loss))
