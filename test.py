from training_routines import train_rgcn, train_gtn
from utils.arguments import model_run_argparse
from torch.utils.tensorboard import SummaryWriter

# edge_index = torch.tensor([[0, 1, 1, 1, 2, 2, 3, 3], [4, 5, 6, 8, 7, 9, 6, 10]])
# node_idx_type = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
# print(hetergeneous_negative_sampling_naive(edge_index, node_idx_type, random_seed=1))

experiment_name = 'GTN40epochs0lambdaFirstRun'
args = model_run_argparse()
writer = SummaryWriter("runs/" + str(experiment_name))

if args.model == 'RGCN':
    output, metadata = train_rgcn(args)
    writer.add_embedding(output, metadata=metadata)
    writer.flush()
    writer.close()
elif args.model == 'GTN':
    output, metadata = train_gtn(args)
    writer.add_embedding(output, metadata=metadata)
    writer.flush()
    writer.close()
else:
    raise NotImplementedError("No implementation for model name: ", args.model)
