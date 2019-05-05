

import datetime
from code_arg import setup_argparse
from code_gd import *
from initialization import *
from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
import time

import copy_reg
import types

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        #print("Bouuuuuuuuuuuund")
        return getattr, (m.im_self, m.im_func.func_name)


#############################################################################
####                               Read data                            #####
#############################################################################
### read for classification purpose
### last column as label (0 or 1), other are features
### for "spambase" we need an extra preprocess step
def read_dataset_file():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = mnist.train.images  # Returns np.array, default [0,1.0], can be changed according to https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
    y = np.asarray(mnist.train.labels, dtype=np.int32)
    print(np.shape(y))
    y = [yi.argmax(axis=0) for yi in y]
    print(np.shape(y))
    x = [xi for xi,yi in zip(x,y) if yi==7 or yi==1]
    y = [yi for yi in y if yi==7 or yi==1]
    y = [1 if yi==1 else -1  for yi in y]
    print(np.shape(y))
    return np.matrix(x), y

# -------------------------------------------------------------------------------
### Select a certain points for train,test and outlier so we have same data for Cross validation
### Should be executed only once (at ONE run of the code)
### then we should read indices from saved file by calling read_one_CV_index()
def store_CV_indices_inFile(size, trnct, tstct, vldct, seed):

    print("number of entire observations: {}".format(size))
    f = open("indexCV_mnist.txt", "w+")
    for i in range(20):
        seed_itr = seed * i + i
        np.random.seed(seed_itr)
        fullperm = np.random.permutation(size)

        sampletrn = fullperm[:trnct]
        sampletst = fullperm[trnct:trnct + tstct]
        samplevld = fullperm[trnct + tstct:trnct + tstct + vldct]

        f.write(','.join([str(val) for val in sampletrn]) + '\n')
        f.write(','.join([str(val) for val in sampletst]) + '\n')
        f.write(','.join([str(val) for val in samplevld]) + '\n')
    f.close()

# -------------------------------------------------------------------------------
### select a certain points for train,test and outlier so we have same data for Cross validation
### we should read indices from saved file
### itr_num: number of the iteration of the program, it depends on us until what iteration we want read data
### in each iteration we should call the function with the corresponding itr_num again (MAX is 20)
### x,y: all initial dataset without any change in indices (but data may just preprocessed)
def read_one_CV_index(x, y, itr_num):

    with open("indexCV_mnist.txt") as f:
        # go forward into the file to reach the saved itr_num indices
        for i in range(itr_num-1):
            for j in range(4):
                f.readline()

        sampletrn = list(map(int, f.readline().split(',')))
        trnx = np.matrix([np.array(x[row]).reshape((x.shape[1],)) for row in sampletrn])
        #print(sampletrn)
        #print(np.shape(y))
        trny = [y[row] for row in sampletrn]

        sampletst = list(map(int, f.readline().split(',')))
        tstx = np.matrix([np.array(x[row]).reshape((x.shape[1],)) for row in sampletst])
        tsty = [y[row] for row in sampletst]

        samplevld = list(map(int, f.readline().split(',')))
        vldx = np.matrix([np.array(x[row]).reshape((x.shape[1],)) for row in samplevld])
        vldy = [y[row] for row in samplevld]

        return trnx, trny, tstx, tsty, vldx, vldy

#############################################################################
####                            Generate results                        #####
#############################################################################

### part of the original main() function that initiate poisoning points and boost
### them accroding to a regression/classification objective function
def generate_poisoning_points(trainx, trainy, testx, testy, validx, validy, poisct):

    types = {'linreg': LinRegGDPoisoner,\
             'lasso': LassoGDPoisoner,\
             'enet': ENetGDPoisoner,\
             'ridge': RidgeGDPoisoner}

    inits = {'rand': rand_my,\
             'initialOutliers': initialOutliers_my}


    init = inits[args.initialization]
    genpoiser = types[args.model](trainx, trainy, testx, testy, validx, validy,
                                  args.eta, args.beta, args.sigma, args.epsilon,
                                  args.multiproc, args.objective)

    err = genpoiser.computeError(genpoiser.initclf)
    auc = genpoiser.computeTestAUC(genpoiser.initclf)

    print("----On clean set----")
    print("Validation MSE: {}".format(err[0]))
    print("Test AUC: {}".format(auc))
    print("Test MSE: {}".format(err[1]))
    clean_set_mse = err[1]
    clean_set_auc = auc

    # initialization step
    poisx, poisy, poisind = init(trainx, trainy, poisct, genpoiser)
    if args.flip == 1:
        poisy = [val * (-1) for val in poisy]
    trainx = np.matrix(trainx)

    clf, _ = genpoiser.learn_model(np.concatenate((trainx, poisx), axis=0), trainy + poisy, None)
    err = genpoiser.computeError(clf)
    auc = genpoiser.computeTestAUC(clf)

    print("----After initialization (flipping?)----")
    print("Validation MSE: {}".format(err[0]))
    print("Test AUC: {}".format(auc))
    print("Test MSE: {}".format(err[1]))
    afterInitialization_mse = err[1]
    afterInitialization_auc = auc

    poiser = types[args.model](trainx, trainy, testx, testy, validx, validy, \
                               args.eta, args.beta, args.sigma, args.epsilon, \
                               args.multiproc, args.objective)
    numsamples = poisct
    curpoisx = poisx[:numsamples, :]
    curpoisy = poisy[:numsamples]

    if args.optimizeAttack == 1:
        poisres, poisresy = poiser.poison_data(curpoisx, curpoisy)
    else:
        poisres, poisresy = poisx, poisy

    poisedx = np.concatenate((trainx, poisres), axis=0)
    poisedy = trainy + poisresy
    clfp, _ = poiser.learn_model(poisedx, poisedy, None)
    err = poiser.computeError(clfp)
    auc = poiser.computeTestAUC(clfp)
    print("----Final attack----")
    print("Validation MSE: {}".format(err[0]))
    print("Test AUC: {}".format(auc))
    print("Test MSE: {}".format(err[1]))
    opt_mse = err[1]
    opt_auc = auc

    return clean_set_mse, afterInitialization_mse, opt_mse, clean_set_auc, afterInitialization_auc, opt_auc

# ------------------------------------------------------------------------------
def main():

    num_itr = args.num_itr
    poisct = int(args.trainct * args.poison_percentage / 100)

    clean_set_mse, flipped_mse, opt_mse, clean_set_auc, flipped_auc, opt_auc = (0,0,0,0,0,0)

    x, y = read_dataset_file()
    for i in range(num_itr):
        print("************** itr {} ****************".format(i+1))
        itr_num = i+1
        trainx, trainy, testx, testy, validx, validy = read_one_CV_index(x, y, itr_num)
        csmse, fmse, opmse , csauc, fauc, opauc = generate_poisoning_points(trainx, trainy, testx, testy, validx, validy, poisct)

        clean_set_mse += csmse
        flipped_mse += fmse
        opt_mse += opmse
        clean_set_auc += csauc
        flipped_auc += fauc
        opt_auc += opauc

    clean_set_mse /= num_itr
    flipped_mse /= num_itr
    opt_mse /= num_itr
    clean_set_auc /= num_itr
    flipped_auc /= num_itr
    opt_auc /= num_itr

    print("***************************************")
    print("************ FINAL MSEs ************")
    print("***************************************")
    print('Clean set MSE: {}'.format(clean_set_mse))
    print('Flipped MSE: {}'.format(flipped_mse))
    print('Optimized MSE: {}'.format(opt_mse))
    print("***************************************")
    print("************ FINAL AUCs ************")
    print("***************************************")
    print('Clean set AUC: {}'.format(clean_set_auc))
    print('Flipped AUC: {}'.format(flipped_auc))
    print('Optimized AUC: {}'.format(opt_auc))


if __name__=='__main__':
    start = time.time()
    parser = setup_argparse()
    args = parser.parse_args()

    print("-----------------------------------------------------------")
    print(args)
    print("-----------------------------------------------------------")

    if (args.read_mode):
        #copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)
        copy_reg.pickle(types.MethodType, _pickle_method)
        main()
    else:
        x, y = read_dataset_file()
        store_CV_indices_inFile(x.shape[0], args.trainct, args.testct, args.validct, args.seed)

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    sys.stdout.flush()




