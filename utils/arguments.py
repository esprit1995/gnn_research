import argparse


def model_run_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ACM',
                        help='which dataset to use. available: any of [ACM, DBLP, IMDB]. Default=ACM')
    parser.add_argument('--model', type=str, default='RGCN',
                        help='which model to use. available: any of [RGCN]. Default=RGCN')
    parser.add_argument('--hidden_dim', type=int, default=50,
                        help='dimensionality of the hidden node representation. Default 50')
    parser.add_argument('--output_dim', type=int, default=50,
                        help='dimensionality of the output node representation. Default 50')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train. Default 100')
    parser.add_argument('--random_seed', type=int, default=10,
                        help='reproducibility: seed for random generators. Default 10')
    parser.add_argument('--type_aware_loss',  action='store_true',
                        help='whether to use type aware loss. Flag')
    parser.add_argument('--type_lambda', type=float, default=1,
                        help='factor of the additional type loss. Only has effect if --type_aware_loss flag is set. Default 1')
    return parser.parse_args()
