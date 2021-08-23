import json

from competitors_perf.baselines.baseline_models import train_metapath2vec, evaluate_ESim_embeddings
from utils.arguments import model_run_argparse

DATASETS = ['ACM', 'DBLP']
PAPERS = ['GTN', 'NSHE']

REDOWNLOAD_DATA = True
EPOCHS = 40
DS_EVAL_FREQ = 5


def pick_best_results(epoch_num: list, metrics: dict) -> dict:
    """
    function that returns best results over the epochs
    for every metric present in metrics dictionary
    :param epoch_num: list containing epoch values for each measurement taken down
    :param metrics: dictionary of format {'metric_name': [metric_val1, metric_val2, ...]}
    :return: dict of format {'metric_name': (best_metric_val, best_metric_val_achieved_at_epoch)}
    """
    result = dict()
    for metric_name in list(metrics.keys()):
        max_val = max(metrics[metric_name])
        max_pos = metrics[metric_name].index(max_val)
        max_at_epoch = epoch_num[max_pos]
        result[metric_name] = (max_val, max_at_epoch)
    return result


if __name__ == "__main__":
    args = model_run_argparse()
    setattr(args, 'redownload_data', REDOWNLOAD_DATA)
    setattr(args, 'epochs', EPOCHS)
    setattr(args, 'downstream_eval_freq', DS_EVAL_FREQ)
    baseline_results = dict()
    for paper in PAPERS:
        setattr(args, 'from_paper', paper)
        for dataset in DATASETS:
            setattr(args, 'dataset', dataset)

            # metapath2vec
            epoch_num, metrics = train_metapath2vec(args)
            baseline_results['metapath2vec_' + str(dataset) + "_" + str(paper)] = pick_best_results(epoch_num, metrics)

            # ESim
            res_dict = evaluate_ESim_embeddings(dataset=(dataset + "_" + paper).lower())
            baseline_results['ESim_' + str(dataset) + '_' + str(paper)] = res_dict[list(res_dict.keys())[0]]

    outfile = open('baseline_results.json', 'w+')
    json.dump(baseline_results, outfile)
    outfile.close()
