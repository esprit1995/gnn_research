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
                    "NSHE": 70}
PAPER_DATASET = {"GTN": ['DBLP'],
                 "NSHE": ['DBLP', 'ACM']}

EXP_NAME_SPECIAL_NOTES = 'test_grid_run'

# ##########################################
# ##########################################

PAPERS_TO_RUN = ["GTN"]
MODELS_TO_RUN = ["GTN"]


#  arguments that affect runs WITH COCLUSTER_LOSS=TRUE
ALTERABLE_ARGS = {'corruption_method': ['random', 'crossover'],
                  'type_lambda': [0.1, 1, 10],
                  'acm_dblp_from_gtn_initial_embs': ['deepwalk', 'original']}


def create_experiment_name(args_, special_notes: str = '') -> str:
    """
    create name for the experiment based on its Argparse settings
    :param args_: Argparse settings of the experiment
    :param special_notes: any non-standard comments to add to the name
    :return:
    """
    if args_.dataset in ['DBLP', 'ACM'] and args_.from_paper == 'GTN':
        dataset_name = '_'.join(
            [args_.dataset, 'from', args_.from_paper, 'initial_embs', args_.acm_dblp_from_gtn_initial_embs])
    else:
        dataset_name = '_'.join([args_.dataset, 'from', args_.from_paper])
    return '_'.join([dataset_name,
                     'Model', args_.model,
                     str(args_.epochs), 'epochs',
                     str(args_.base_triplet_loss), 'baseTripletLoss',
                     str(args_.cocluster_loss), 'coclusterLoss',
                     str(args_.type_lambda), 'lambda',
                     str(args_.corruption_method), 'corrmethod',
                     str(args_.instances_per_template), 'ipt',
                     str(args_.random_seed), 'rs',
                     datetime.now().strftime("%d_%m_%Y_%H_%M_%S"),
                     special_notes])


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

                cprint('-->Training without cocluster loss', color='yellow')
                setattr(args, 'cocluster_loss', False)
                GTN_initial_embs = ['original']
                if from_paper == 'GTN':
                    GTN_initial_embs.append('deepwalk')
                for gtn_ie in GTN_initial_embs:
                    setattr(args, 'acm_dblp_from_gtn_initial_embs', gtn_ie)
                    p = mp.Process(target=run_pipeline, args=(args, create_experiment_name(args, EXP_NAME_SPECIAL_NOTES)))
                    p.start()
                    p.join()

                cprint('-->Training with cocluster loss', color='yellow')
                setattr(args, 'cocluster_loss', True)
                # GTN datasets have different initial embeddings possibilities
                # others don't, have to account for that.
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
                    p = mp.Process(target=run_pipeline,
                                   args=(args, create_experiment_name(args, EXP_NAME_SPECIAL_NOTES)))
                    p.start()
                    p.join()
