import argparse


def setup_argparse():
    parser = argparse.ArgumentParser(description='handle poisoning inputs')

    # counts
    parser.add_argument("-r", "--trainct", default=300, type=int, help='number of points to train models with')
    parser.add_argument("-t", "--testct", default=1000, type=int, help='number of points to test models on')
    parser.add_argument("-v", "--validct", default=10, type=int, help='size of validation set')
    parser.add_argument("-pois_per", "--poison_percentage", type=int, default=20, help='poisoning data percentage')

    # params for gd poisoning
    parser.add_argument("-l", "--lambd", default=1, type=float, help='lambda value to use in poisoning;icml 2015')
    parser.add_argument("-n", "--epsilon", default=1e-3, type=float, help='termination condition epsilon;icml 2015')
    parser.add_argument("-a", "--eta", default=0.005, type=float, help='line search multiplier eta; icml 2015')
    parser.add_argument("-b", "--beta", default=0.05, type=float, help='line search decay beta; icml 2015')
    parser.add_argument("-i", "--sigma", default=0.9, type=float, help='line search termination lowercase sigma;icml 2015')

    # store_false: Read, store_true: Write
    parser.add_argument("-rmode", '--read_mode', action='store_false', help='to "Write" Cross Val data into file or "Read and Execute" it')
    # init strategy
    parser.add_argument('-init', '--initialization', default='initialOutliers',choices=['rand' , 'initialOutliers'])
    parser.add_argument("-itr", "--num_itr", type=int, default=1, help='num of iteration program repeats')


    # objective  --- Wtr or Wval (outer obj function in bi-level opt)
    parser.add_argument("-obj", "--objective", default=1, type=int, help="objective to use (0 for train, 1 for validation)")
    # seed for randmization
    parser.add_argument('-seed', type=int, default=123, help='random seed')
    # enable multi processing, store_true: sequential
    parser.add_argument("-mp", "--multiproc", action='store_false', help='enable to allow for multiprocessing support')

    parser.add_argument("-m", "--model", default='linreg', choices=['linreg', 'lasso', 'enet', 'ridge'], \
                        help="choose linreg for linear regression, lasso for lasso, enet for elastic net, or ridge for ridge")

    parser.add_argument("-flip", "--flip", type=int, default=1, help='Whether flip the poisoning points. flip:1, not flip:0')
    parser.add_argument("-optAttack", "--optimizeAttack", type=int, default=1, help='Whether optimize the poisoning points(features). opt:1, not opt:0')

    return parser

