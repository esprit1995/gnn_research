import os
import warnings
import torch.multiprocessing as mp
from datetime import datetime
from termcolor import cprint
from pathlib import Path
from run_pipeline import run_pipeline
from create_experiment_excel import collect_model_results
from utils.arguments import model_run_argparse

MODELS_TO_RUN = ['RGCN']
DATASETS_TO_RUN = ['DBLP', 'ACM']
EPOCHS_MAX = [500] # [100, 150, 200, 250, 300]

EXCEL_PATH = os.path.join(os.getcwd(), 'data', 'comparative_excels')
EXCEL_NAME = 'comparative_table.xlsx'

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    args = model_run_argparse()
    for model in MODELS_TO_RUN:
        setattr(args, 'redownload_data', True)
        setattr(args, 'cocluster_loss', True)
        setattr(args, 'base_triplet_loss', False)
        setattr(args, 'model', model)
        for epochs in EPOCHS_MAX:
            setattr(args, 'epochs', epochs)
            for dataset in DATASETS_TO_RUN:
                setattr(args, 'dataset', dataset)
                cprint('Running ' + str(model) + ' on ' + str(dataset) + ' for ' + str(args.epochs) + ' epochs',
                       color='cyan',
                       attrs=['bold'])
                special_notes = 'added_regularization'
                experiment_name = '_'.join([args.dataset,
                                            args.model,
                                            str(args.epochs), 'epochs',
                                            str(args.base_triplet_loss), 'baseTripletLoss',
                                            str(args.cocluster_loss), 'coclusterLoss',
                                            str(args.type_lambda), 'lambda',
                                            str(args.corruption_method), 'corrmethod',
                                            str(args.instances_per_template), 'ipt',
                                            str(args.random_seed), 'rs',
                                            datetime.now().strftime("%d_%m_%Y_%H:%M:%S"),
                                            special_notes])
                p = mp.Process(target=run_pipeline, args=(args, experiment_name))
                p.start()
                p.join()
        warnings.simplefilter('ignore')
        model_experiment_df = collect_model_results(model, experiments_path=os.path.join(os.getcwd(),
                                                                                         'experiment_records'))
        if not os.path.isdir(EXCEL_PATH):
            Path(EXCEL_PATH).mkdir(parents=True, exist_ok=True)
        model_experiment_df.to_excel(os.path.join(EXCEL_PATH,
                                                  '_'.join([str(model), EXCEL_NAME])))
        warnings.simplefilter('default')
