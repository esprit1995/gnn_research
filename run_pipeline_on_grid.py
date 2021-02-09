import os
import warnings
import torch.multiprocessing as mp
from termcolor import cprint
from pathlib import Path
from test import run_pipeline
from create_experiment_excel import collect_model_results
from utils.arguments import model_run_argparse

MODELS_TO_RUN = ['RGCN']
DATASETS_TO_RUN = ['DBLP', 'ACM', 'IMDB']
EPOCHS_MAX = [800]

TYPE_AWARE = False
TYPE_LAMBDA = 0

EXCEL_PATH = os.path.join(os.getcwd(), 'data', 'comparative_excels')
EXCEL_NAME = 'comparative_table.xlsx'

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    args = model_run_argparse()
    for model in MODELS_TO_RUN:
        setattr(args, 'model', model)
        setattr(args, 'epochs', EPOCHS_MAX[MODELS_TO_RUN.index(model)])
        for dataset in DATASETS_TO_RUN:
            setattr(args, 'dataset', dataset)
            cprint('Running ' + str(model) + ' on ' + str(dataset) + ' for ' + str(args.epochs) + ' epochs',
                   color='cyan',
                   attrs=['bold'])
            p = mp.Process(target=run_pipeline, args=(args,))
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
