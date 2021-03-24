import argparse


def str2bool(string):
    trues = ['1', 'true', 'True', 'yes', 'Yes']
    if str(string) in trues:
        return True
    else:
        return False


def model_run_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ACM',
                        help='which dataset to use. available: any of [ACM, DBLP, IMDB, ACM_HAN]. Default=ACM')
    parser.add_argument('--model', type=str, default='RGCN',
                        help='which model to use. available: any of [RGCN, GTN]. Default=RGCN')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train. Default 100')
    parser.add_argument('--random_seed', type=int, default=10,
                        help='reproducibility: seed for random generators. Default 10')
    parser.add_argument('--type_aware_loss', action='store_true',
                        help='whether to use type aware loss. Flag')
    parser.add_argument('--cocluster_loss', action='store_true',
                        help='whether to use additional co-clustering loss')
    parser.add_argument('--corruption_method', type=str, default='random',
                        help='method of positive instance corruption')
    parser.add_argument('--multitype_labels', type=str2bool, default=True,
                        help='whether to use multitype labels for ACM/DBLP')
    parser.add_argument('--type_lambda', type=float, default=1,
                        help='factor of the additional type loss. Only has effect if --type_aware_loss flag is set. Default 1')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate for the model')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='l2 reg')
    parser.add_argument('--device', type=str, default='cpu',
                        help='device to run the model on - cpu or cuda. Default cpu')

    return parser.parse_args()
