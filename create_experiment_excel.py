import argparse
import re
import os
import pandas as pd
import json
import warnings

from termcolor import cprint
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True,
                    help='name of the architecture to collect experiments for')
parser.add_argument('--experiments_path', type=str, default=os.path.join(os.getcwd(), 'experiment_records'),
                    help='full path to the folder containing the experiment results')

parser.add_argument('--save_to', type=str, default=os.path.join(os.getcwd(), 'data', 'comparative_excels'),
                    help='directory to save the results to')
parser.add_argument('--save_filename', type=str, default='aggregated_excel.xlsx',
                    help='name under which the file will be saved')


def collect_model_results(model_name: str, experiments_path: str,
                          dataset: str = None, epochs: int = None, ) -> pd.DataFrame:
    """
    collect performance metrics of a given model, return them in a DataFrame format
    :param model_name:  name of the architecture to collect experiments for
    :param experiments_path: full path to the folder containing the experiment results
    :param dataset: optionally, name of the dataset to collect the results for
    :param epochs: optionally, number of epochs the net was trained for
    :return: pd.DataFrame containing the perf metrics. Currently: nmi, rmi, microf1, macrof1
    """
    if not os.path.isdir(experiments_path):
        cprint('collect_model_results()::experiments_path argument faulty, please verify', color='red', attrs=['bold'])
        raise ValueError()
    relevant_dirs = list()
    for filename in os.listdir(experiments_path):
        elements = str(filename).split('_')
        if elements[1] != model_name:
            continue
        if dataset is not None and dataset != elements[0]:
            continue
        if epochs is not None and epochs != int(elements[2]):
            continue
        relevant_dirs.append(filename)
    cprint('#relevant dirs found: ' + str(len(relevant_dirs)), color='blue')

    result = pd.DataFrame(columns=['model', 'dataset', 'epochs',
                                   'argparse_args',
                                   'type_aware', 'type_lambda',
                                   'NMI', 'ARI',
                                   'microF1', 'macroF1'])
    for dirname in relevant_dirs:
        directory = os.path.join(experiments_path, dirname)
        argparse_filename = [elem for elem in os.listdir(directory) if re.match('argparse_*', elem)][0]
        metrics_filename = [elem for elem in os.listdir(directory) if re.match('perf_metrics*', elem)][0]
        argparse_args = json.load(open(os.path.join(directory, argparse_filename), 'r'))
        metrics = json.load(open(os.path.join(directory, metrics_filename), 'r'))

        new_row = pd.DataFrame({'model': model_name,
                                'dataset': dirname.split('_')[0],
                                'epochs': int(dirname.split('_')[2]),
                                'argparse_args': json.dumps(argparse_args),
                                'type_aware': bool(argparse_args['type_aware_loss']),
                                'type_lambda': argparse_args['type_lambda'],
                                'NMI': metrics['NMI'],
                                'ARI': metrics['ARI'],
                                'microF1': metrics['microF1'],
                                'macroF1': metrics['macroF1']}, index=[0])
        result = result.append(new_row)
    result_final = pd.DataFrame(columns=['model', 'dataset', 'epochs',
                                         'argparse_args',
                                         'type_aware', 'type_lambda',
                                         'NMI', 'ARI',
                                         'microF1', 'macroF1',
                                         'NMI_gain', 'ARI_gain', 'microF1_gain', 'macroF1_gain'])
    for _, group in result.groupby(['model', 'dataset', 'epochs']):
        baseline = group[group['type_aware'] == False][['NMI', 'ARI', 'microF1', 'macroF1']].reset_index(drop=True)
        for col in ['NMI', 'ARI', 'microF1', 'macroF1']:
            group[col + '_gain'] = group[col].apply(lambda x: (x - baseline.loc[0, col]) / baseline.loc[0, col] * 100)
        result_final = result_final.append(group)
    result_final = result_final.sort_values(by=['model', 'dataset', 'epochs', 'type_aware', 'type_lambda'])
    for col in ['NMI', 'ARI', 'microF1', 'macroF1', 'NMI_gain', 'ARI_gain', 'microF1_gain', 'macroF1_gain']:
        result_final[col] = result_final[col].apply(lambda x: round(float(x), 3))
    return result_final.reset_index(drop=True)


if __name__ == "__main__":
    warnings.simplefilter('ignore')
    args = parser.parse_args()
    experiment_df = collect_model_results(args.model, args.experiments_path)
    if not os.path.isdir(args.save_to):
        Path(args.save_to).mkdir(parents=True, exist_ok=True)
    experiment_df.to_excel(os.path.join(args.save_to, args.save_filename))
    warnings.simplefilter('default')
