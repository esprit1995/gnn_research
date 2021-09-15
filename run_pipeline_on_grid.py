import torch.multiprocessing as mp
from datetime import datetime
from termcolor import cprint
from run_pipeline import run_pipeline
from utils.arguments import model_run_argparse
from sklearn.model_selection import ParameterGrid


# ***************##########################################
#  Experimental settings
# ##########################################***************

# these ones have been used in the results currently presented in the thesis
# COCLUSTERING_METAPATHS = {'ACM': [('0', '1', '0', '2'), ('2', '0', '1')],
#                           'DBLP': [('0', '1', '2'), ('0', '1', '0'), ('1', '2', '1')]}
# CORRUPTION_POSITIONS = {'ACM': [(1, 2), (2, 2)],
#                         'DBLP': [(1, 1), (1, 2), (2, 2)]}

# trying to get a difference between corruption methods to be more apparent.
# COCLUSTERING_METAPATHS = {'ACM': [('1', '0', '2', '0', '1')],
#                           'DBLP': [('0', '1', '2', '1', '0')]}
# CORRUPTION_POSITIONS = {'ACM': [(1, 3)],
#                         'DBLP': [(2, 4)]}

MODEL_EVAL_FREQ = {"RGCN": 5,
                   "GTN": 2,
                   "NSHE": 2,
                   "MAGNN": 1,
                   "HeGAN": 1,
                   "VGAE": 3}

MODEL_MAX_EPOCHS = {'RGCN': 500,
                    'GTN': 35,
                    "NSHE": 100,
                    "MAGNN": 15,
                    "HeGAN": 15,
                    "VGAE": 200}

PAPER_DATASET = {"GTN": ['DBLP', 'ACM'],
                 "NSHE": ['ACM', 'DBLP']}

EXP_NAME_SPECIAL_NOTES = 'combine_method_switch'

# ##########################################
# ##########################################

PAPERS_TO_RUN = ["NSHE", 'GTN']
MODELS_TO_RUN = ["HeGAN"]


#  arguments that affect runs WITH COCLUSTER_LOSS=TRUE
ALTERABLE_ARGS = {'corruption_method': ['random'],
                  'loss_combine_method': ['scaled'],
                  'acm_dblp_from_gtn_initial_embs': ['deepwalk'],
                  'homogeneous_VGAE': [True, False]}


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

                # cprint('-->Training without cocluster loss', color='yellow')
                # setattr(args, 'cocluster_loss', False)
                # GTN_initial_embs = ['original']
                # homogeneous_VGAE = [True]
                # if from_paper == 'GTN':
                #     GTN_initial_embs.append('deepwalk')
                # if model == 'VGAE':
                #     homogeneous_VGAE.append(False)
                # for gtn_ie in GTN_initial_embs:
                #     setattr(args, 'acm_dblp_from_gtn_initial_embs', gtn_ie)
                #     for vgae_mode in homogeneous_VGAE:
                #         setattr(args, 'homogeneous_VGAE', vgae_mode)
                #         p = mp.Process(target=run_pipeline, args=(args, create_experiment_name(args, EXP_NAME_SPECIAL_NOTES)))
                #         p.start()
                #         p.join()


                cprint('-->Training with cocluster loss', color='yellow')
                setattr(args, 'cocluster_loss', True)
                # GTN datasets have different initial embeddings possibilities
                # others don't, have to account for that.
                if from_paper == 'GTN':
                    altags = ALTERABLE_ARGS
                else:
                    altags = {argname: ALTERABLE_ARGS[argname] for argname in list(ALTERABLE_ARGS.keys()) if
                              argname != 'acm_dblp_from_gtn_initial_embs'}
                # VGAE model has 2 variants: homogeneous, heterogeneous
                if model != 'VGAE':
                    altags = {argname: ALTERABLE_ARGS[argname] for argname in list(ALTERABLE_ARGS.keys()) if
                              argname != 'homogeneous_VGAE'}

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
