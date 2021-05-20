import argparse
import re
import os
import pandas as pd
import json
import warnings

from termcolor import cprint
from pathlib import Path

AVAILABLE_MODELS = ['RGCN', 'NSHE', 'GTN']

parser = argparse.ArgumentParser()
parser.add_argument('--experiments_path', type=str, default=os.path.join(os.getcwd(), 'experiment_records'),
                    help='full path to the folder containing the experiment results')

parser.add_argument('--save_to', type=str, default=os.path.join(os.getcwd(), 'data', 'comparative_excels'),
                    help='directory to save the results to')
parser.add_argument('--save_filename', type=str, default='aggregated_excel.xlsx',
                    help='name under which the file will be saved')


def collect_model_results(model_name: str,
                          experiments_path: str = '/home/ubuntu/msandal_code/PyG_playground/experiment_records/',
                          dataset: str = None, from_paper: str = None) -> pd.DataFrame:
    """
    collect performance metrics of a given model, return them in a DataFrame format
    :param model_name:  name of the architecture to collect experiments for
    :param experiments_path: full path to the folder containing the experiment results
    :param dataset: optionally, name of the dataset to collect the results for
    :param from_paper: name of the paper from which the dataset came from
    :return: pd.DataFrame containing the perf metrics. Currently: nmi, rmi, microf1, macrof1
    """
    warnings.simplefilter('ignore')
    if not os.path.isdir(experiments_path):
        cprint('collect_model_results()::experiments_path argument faulty, please verify', color='red', attrs=['bold'])
        raise ValueError()
    relevant_dirs = list()
    ds_name = None
    if dataset is not None:
        if from_paper is None:
            raise ValueError(
                'collect_model_results(): if dataset is specified, then "from_paper" arg must also be specified')
        ds_name = str(dataset) + '_from_' + str(from_paper)
    model = 'Model_' + str(model_name).upper()
    cprint('---> collecting experimental results for model ' + str(model_name),
           color='cyan')
    if ds_name is not None:
        cprint('on dataset ' + str(ds_name), color='cyan')
    for filename in os.listdir(experiments_path):
        if model not in str(filename):
            continue
        if ds_name is not None and ds_name not in filename:
            continue
        relevant_dirs.append(filename)

    cprint('#relevant dirs found: ' + str(len(relevant_dirs)), color='cyan')

    result = pd.DataFrame(columns=['model', 'dataset', 'epochs',
                                   'argparse_args',
                                   'cocluster_loss', 'lambda',
                                   'NMI', 'ARI',
                                   'microF1', 'macroF1', 'best_NMI', 'best_NMI_epoch'])
    for dirname in relevant_dirs:
        directory = os.path.join(experiments_path, dirname)
        argparse_filename = [elem for elem in os.listdir(directory) if re.match('argparse_*', elem)][0]
        metrics_filename = [elem for elem in os.listdir(directory) if re.match('perf_metrics*', elem)][0]
        argparse_args = json.load(open(os.path.join(directory, argparse_filename), 'r'))
        metrics = json.load(open(os.path.join(directory, metrics_filename), 'r'))

        new_row = pd.DataFrame({'model': argparse_args['model'],
                                'dataset': '_'.join([argparse_args['dataset'], argparse_args['from_paper']]),
                                'initial_embs_GTN_paper': argparse_args["acm_dblp_from_gtn_initial_embs"],
                                'epochs': argparse_args['epochs'],
                                'argparse_args': json.dumps(argparse_args),
                                'cocluster_loss': bool(argparse_args['cocluster_loss']),
                                'lambda': argparse_args['type_lambda'],
                                'corruption_method': argparse_args['corruption_method'],
                                'NMI': metrics['NMI'],
                                'ARI': metrics['ARI'],
                                'microF1': metrics['microF1'],
                                'macroF1': metrics['macroF1'],
                                'best_NMI': metrics['best_NMI'],
                                'best_NMI_epoch': metrics['best_NMI_at_epoch']}, index=[0])
        result = result.append(new_row)

    #  if cocluster loss == False, other parameters don't matter.
    #  that's why we drop duplicates like follows:
    result = pd.concat([result.query('cocluster_loss==False') \
                       .drop_duplicates(['model', 'dataset', 'initial_embs_GTN_paper', 'cocluster_loss']),
                        result.query('cocluster_loss==True')])
    result_final = pd.DataFrame(columns=['model', 'dataset', 'initial_embs_GTN_paper', 'epochs',
                                         'argparse_args',
                                         'cocluster_loss', 'lambda', 'corruption_method',
                                         'NMI', 'ARI',
                                         'microF1', 'macroF1', 'best_NMI', 'best_NMI_epoch',
                                         'best_NMI_gain'])
    for _, group in result.groupby(['model', 'dataset', 'initial_embs_GTN_paper']):
        baseline = group[group['cocluster_loss'] == False][['best_NMI']].reset_index(drop=True)
        for col in ['best_NMI']:
            group[col + '_gain'] = group[col].apply(lambda x: (x - baseline.loc[0, col]) / baseline.loc[0, col] * 100)
        result_final = result_final.append(group)
    result_final = result_final.sort_values(by=['model', 'dataset', 'initial_embs_GTN_paper',
                                                'cocluster_loss', 'lambda'])
    for col in ['NMI', 'ARI', 'microF1', 'macroF1', 'best_NMI_gain', 'best_NMI']:
        result_final[col] = result_final[col].apply(lambda x: round(float(x), 3))
    warnings.simplefilter('default')
    return result_final.reset_index(drop=True)


def collect_all_results(experiments_path: str = '/home/ubuntu/msandal_code/PyG_playground/experiment_records/',
                        dataset: str = None, from_paper: str = None) -> pd.DataFrame:
    """
    collect all results available in the specified directory and return them as a
    pandas.DataFrame
    :param experiments_path: where to look for recorded results
    :param dataset: for which dataset to collect the results. If None, all available datasets are collected
    :param from_paper: paper from which the dataset (see above) came from. Must be specified if dataset arg is not
                       None. Else doesn't matter.
    """
    all_results = None
    for model in AVAILABLE_MODELS:
        if all_results is None:
            all_results = collect_model_results(model_name=model,
                                                experiments_path=experiments_path,
                                                dataset=dataset,
                                                from_paper=from_paper)
        else:
            all_results = pd.concat([all_results, collect_model_results(model_name=model,
                                                                        experiments_path=experiments_path,
                                                                        dataset=dataset,
                                                                        from_paper=from_paper)])
    return all_results


if __name__ == "__main__":
    warnings.simplefilter('ignore')
    args = parser.parse_args()
    experiment_df = collect_all_results(experiments_path=args.experiments_path)
    if not os.path.isdir(args.save_to):
        Path(args.save_to).mkdir(parents=True, exist_ok=True)
    experiment_df.to_excel(os.path.join(args.save_to, args.save_filename))
    warnings.simplefilter('default')
