import torch.multiprocessing as mp
from datetime import datetime
from termcolor import cprint
from run_pipeline import run_pipeline
from utils.arguments import model_run_argparse
from sklearn.model_selection import ParameterGrid

MODEL_EVAL_FREQ = {"RGCN": 5,
                   "GTN": 2,
                   "NSHE": 2}
MODEL_MAX_EPOCHS = {'RGCN': 500,
                    'GTN': 30,
                    "NSHE": 50}
PAPER_DATASET = {"GTN": ['DBLP', 'ACM'],
                 "NSHE": ['DBLP', 'ACM']}

# ##########################################
# ##########################################

PAPERS_TO_RUN = ['GTN', 'NSHE']
MODELS_TO_RUN = ["NSHE", 'GTN']

ALTERABLE_ARGS = {'cocluster_loss': [True, False],
                  'corruption_method': ['random', 'crossover'],
                  'type_lambda': [0.1, 1, 10],
                  'acm_dblp_from_gtn_initial_embs': ['deepwalk', 'original']}

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    args = model_run_argparse()

    for model in MODELS_TO_RUN:
        for from_paper in PAPERS_TO_RUN:
            setattr(args, 'from_paper', from_paper)
            setattr(args, 'redownload_data', True)
            setattr(args, 'base_triplet_loss', True)
            setattr(args, 'model', model)
            setattr(args, 'epochs', MODEL_MAX_EPOCHS[model])
            setattr(args, 'downstream_eval_freq', MODEL_EVAL_FREQ[model])
            for dataset in PAPER_DATASET[from_paper]:
                setattr(args, 'dataset', dataset)
                cprint('Running ' + str(model) + ' on ' + str(dataset) + ' from paper ' + str(from_paper)
                       + ' for ' + str(args.epochs) + ' epochs',
                       color='cyan',
                       attrs=['bold'])
                if from_paper == 'GTN':
                    altags = ALTERABLE_ARGS
                else:
                    altags = {argname: ALTERABLE_ARGS[argname] for argname in list(ALTERABLE_ARGS.keys()) if
                                                                argname != 'acm_dblp_from_gtn_initial_embs'}
                alterable_args_grid = ParameterGrid(altags)
                for alterable_args_set in alterable_args_grid:
                    cprint('alterable args values: ' + str(alterable_args_set),
                           color='yellow')
                    # set the values of all the alterable arguments
                    for attrname in list(alterable_args_set.keys()):
                        setattr(args, attrname, alterable_args_set[attrname])
                    # create experiment name
                    special_notes = ''
                    if args.dataset in ['DBLP', 'ACM'] and args.from_paper == 'GTN':
                        dataset_name = '_'.join(
                            [args.dataset, 'from', args.from_paper, 'initial_embs', args.acm_dblp_from_gtn_initial_embs])
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
                    p = mp.Process(target=run_pipeline, args=(args, experiment_name))
                    p.start()
                    p.join()
