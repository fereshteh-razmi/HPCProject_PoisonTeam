from __future__ import division, print_function

import datetime
from code_arg import setup_argparse
from initialization import *
from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
import time
import types


# functional
import random
import numpy as np
import numpy.linalg as la
import argparse
import sys

# visualization
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# sklearn imports
from sklearn.metrics import roc_auc_score
from sklearn import linear_model

import time


def poison_data(trnx, trny, tstx, tsty, vldx, vldy, poisx, poisy):
    """
    poison_data takes an initial set of poisoning points and optimizes it
    using gradient descent with parameters set in __init__

    poisxinit, poisyinit: initial poisoning points
    tstart: start time - used for writing out performance
    visualize: whether we want to visualize the gradient descent steps
    newlogdir: directory to log into, to save the visualization
    """

    poisct = poisx.shape[0]

    best_poisx = np.zeros(poisx.shape)
    best_poisy = [None for a in poisy]

    best_obj = -1
    count = 0
    print("before pooling")
    if args.multiproc:
        import multiprocessing as mp
        print("after importing")
        workerpool = 1
    else:
        workerpool = 0#None

    print("after pooling")
    sig = compute_sigma(trnx)  # can already compute sigma and mu
    mu = compute_mu(trnx)  # as x_c does not change them
    eq7lhs = np.bmat([[sig, np.transpose(mu)],
                      [mu, np.matrix([1])]])

    # figure out starting error
    it_res = iter_progress(trnx, trny, tstx, tsty, vldx, vldy, poisx, poisy, poisx, poisy)

    if it_res[0] > best_obj:
        best_poisx, best_poisy, best_obj = poisx, poisy, it_res[0]

    # main work loop
    best_count = 0 ####
    best_res = it_res[0]
    pool = None
    if workerpool == 1: 
        pool = mp.Pool()
        print("pooooool number: "+str(mp.cpu_count()))


    while True:
        start = time.time()
        count += 1
        print("count =" + str(count)) #######
        new_poisx = np.matrix(np.zeros(poisx.shape))
        new_poisy = [None for a in poisy]
        x_cur = np.concatenate((trnx, poisx), axis=0)
        y_cur = trny + poisy

        samplenum = trnx.shape[0]
        feanum = trnx.shape[1]
        clf, lam = learn_model(x_cur, y_cur, None) #trnx_short, trny_short, vldx_short, vldy_short) \
        pois_params = [(poisx[i], poisy[i], eq7lhs, mu, clf, lam, samplenum, feanum, vldx, vldy) \
                       for i in range(poisct)]
        outofboundsct = 0

        if workerpool == 1:

            for i, cur_pois_res in enumerate(
                     pool.map(poison_data_subroutine,pois_params)):
                new_poisx[i] = cur_pois_res[0]
                new_poisy[i] = cur_pois_res[1]
                outofboundsct += cur_pois_res[2]

        else:
            for i in range(poisct):
                cur_pois_res = poison_data_subroutine(pois_params[i])
                new_poisx[i] = cur_pois_res[0]
                new_poisy[i] = cur_pois_res[1]
                outofboundsct += cur_pois_res[2]

        it_res = iter_progress(trnx, trny, tstx, tsty, vldx, vldy, poisx, poisy, new_poisx, new_poisy)
        # if we don't make progress, decrease learning rate
        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
        sys.stdout.flush()
        if (it_res[0] <= it_res[1]):
            args.eta *= 0.75
            poisx, poisy = new_poisx, new_poisy ###### added
        else:
            best_count = count
            best_res = it_res[0]
            poisx = new_poisx
            poisy = new_poisy


        if (it_res[0] > best_obj):
            best_poisx, best_poisy, best_obj = poisx, poisy, it_res[0] ### changed it_res[1] to it_res[0]

        it_diff = abs(it_res[0] - it_res[1])

        # stopping conditions
        if (count >= 5 and (it_diff <= args.epsilon or count > 5)):
            break

    if workerpool == 1:
        pool.close()

    return best_poisx, best_poisy


def poison_data_subroutine(in_tuple):
    """
    poison_data_subroutine poisons a single poisoning point
    input is passed in as a tuple and immediately unpacked for
    use with the multiprocessing.Pool.map function

    poisxelem, poisyelem: poisoning point at the start
    eq7lhs, mu: values for computation
    clf, lam: current model and regularization coef
    """
    poisxelem, poisyelem, eq7lhs, mu, clf, lam , samplenum, feanum, vldx, vldy = in_tuple
    m = compute_m(clf, poisxelem, poisyelem, feanum)
    feanum = poisxelem.shape[1]
    
    # compute partials
    wxc, bxc, wyc, byc = compute_wb_zc(eq7lhs, mu, clf.coef_, m, \
                                            samplenum, poisxelem)


    otherargs = None
    attack, attacky = comp_attack_vld(clf, wxc, bxc, wyc, byc, otherargs, vldx, vldy)

    # keep track of how many points are pushed out of bounds
    if (poisyelem >= 1 and attacky >= 0) \
            or (poisyelem <= 0 and attacky <= 0):
        outofbounds = True
    else:
        outofbounds = False

    # include y in gradient normalization
    allattack = attack.ravel()

    norm = np.linalg.norm(allattack)
    allattack = allattack / norm if norm > 0 else allattack
    attack = allattack

    poisxelem, poisyelem, _ = lineSearch(vldx, vldy,poisxelem, poisyelem, \
                                              attack, attacky, clf, lam)
    poisxelem = poisxelem.reshape((1, feanum))

    return poisxelem, poisyelem, outofbounds


def lineSearch(vldx, vldy, poisxelem, poisyelem, attack, attacky, clf, lam):
    k = 0
    curx = poisxelem
    cury = [poisyelem]
    clf1, lam1 = clf, lam
    clf1.partial_fit(curx,cury)

    lastpoisxelem = poisxelem
    curpoisxelem = poisxelem

    lastyc = poisyelem
    curyc = poisyelem
    otherargs = None

    w_1 = comp_obj_vld(clf1, lam1, otherargs, vldx, vldy)
    count = 0
    eta = args.eta

    itr = 0
    while True:
        itr = itr + 1
        if (count > 0):
            eta = args.beta * eta
        count += 1
        curpoisxelem = curpoisxelem + eta * attack
        curpoisxelem = np.clip(curpoisxelem, 0, 1)
        curx = curpoisxelem
        clf1, lam1 = clf, lam
        clf1.partial_fit(curx,cury)
        w_2 = comp_obj_vld(clf1, lam1, otherargs, vldx, vldy)

        if (count >= 5000):
            break

        lastpoisxelem = curpoisxelem
        lastyc = curyc
        w_1 = w_2
        k += 1

    curx = np.delete(curx, curx.shape[0] - 1, axis=0)
    curx = np.append(curx, curpoisxelem, axis=0)
    cury[-1] = curyc
    clf1, lam1 = learn_model(curx, cury, None)

    w_2 = comp_obj_vld(clf1, lam1, otherargs, vldx, vldy)

    return np.clip(curpoisxelem, 0, 1), curyc, w_2

def computeTestAUC(tstx, tsty, clf):
    y_score = clf.predict(tstx)
    y_true = tsty
    return roc_auc_score(y_true, y_score)

def computeError(tstx, tsty, vldx, vldy, clf):
    toterr, v_toterr = 0, 0
    rsqnum, v_rsqnum = 0, 0
    rsqdenom, v_rsqdenom = 0, 0

    feanum = tstx.shape[1]
    w = np.reshape(clf.coef_, (feanum,))
    sum_w = np.linalg.norm(w, 1)

    mean = sum(tsty) / len(tsty)
    vmean = sum(vldy) / len(vldy)

    pred = clf.predict(tstx)
    vpred = clf.predict(vldx)

    for i, trueval in enumerate(vldy):
        guess = vpred[i]
        if guess < 0 :
            guess = -1
        else:
            guess = 1
        err = guess - trueval

        v_toterr += err ** 2  # MSE
        v_rsqnum += (guess - vmean) ** 2  # R^2 num and denom
        v_rsqdenom += (trueval - vmean) ** 2

    for i, trueval in enumerate(tsty):
        guess = pred[i]
        if guess < 0 :
            guess = -1
        else:
            guess = 1

        err = guess - trueval

        toterr += err ** 2  # MSE
        rsqnum += (guess - mean) ** 2  # R^2 num and denom
        rsqdenom += (trueval - mean) ** 2

    vld_mse = v_toterr / len(vldy)
    tst_mse = toterr / len(tsty)

    return vld_mse, tst_mse



def iter_progress(trnx, trny, tstx, tsty, vldx, vldy, lastpoisx, lastpoisy, curpoisx, curpoisy):
    x0 = np.concatenate((trnx, lastpoisx), axis = 0)
    y0 = trny + lastpoisy
    clf0, lam0 = learn_model(x0, y0, None)
    w_0 = comp_obj_vld(clf0, lam0, None, vldx, vldy)

    x1 = np.concatenate((trnx, curpoisx), axis = 0)
    y1 = trny + curpoisy
    clf1, lam1 = learn_model(x1, y1, None)
    w_1 = comp_obj_vld(clf1, lam1, None, vldx, vldy)
    err = computeError(tstx, tsty, vldx, vldy, clf1)

    return w_1, w_0, err


############################################################################################
# Implements GD Poisoning for OLS Linear Regression
############################################################################################



def learn_model( x, y, clf):
    if (not clf):
        #clf = linear_model.Ridge(alpha=0.00001) ###### original value is 0.00001
        clf = linear_model.SGDRegressor(loss="squared_loss", penalty=None)
        # clf = linear_model.LinearRegression()
        #print("iter_n:"+str(clf.n_iter))
        #print("len(y):"+str(len(y)))
        #clf.n_iter = np.ceil(10**6 / len(y))

    clf.fit(x, y)
    return clf, 0

def compute_sigma(trnx):
    sigma = np.dot(np.transpose(trnx), trnx)
    sigma = sigma / trnx.shape[0]
    return sigma

def compute_mu(trnx):
    mu = np.mean(trnx, axis=0)
    return mu

def compute_m(clf, poisxelem, poisyelem, feanum):
    w,b = clf.coef_, clf.intercept_
    poisxelemtransp = np.reshape(poisxelem, (feanum,1) )
    wtransp = np.reshape(w, (1,feanum) )
    errterm = (np.dot(w, poisxelemtransp) + b - poisyelem).reshape((1,1))
    first = np.dot(poisxelemtransp,wtransp)
    m = first + errterm[0,0]*np.identity(feanum)
    return m

def compute_wb_zc(eq7lhs, mu, w, m, n, poisxelem):
    eq7rhs = -(1 / n) * np.bmat([[m, -np.matrix(poisxelem.T)],
                                 [np.matrix(w.T), np.matrix([-1])]])

    ###### in previous versions of lstsq last argument rcond should be float with defualt -1 #####
    ###### here they put it None (according to new version) and got error, I removed it #####
    wbxc = np.linalg.lstsq(eq7lhs, eq7rhs,rcond=-1)[0]
    wxc = wbxc[:-1, :-1]  # get all but last row
    bxc = wbxc[-1, :-1]  # get last row
    wyc = wbxc[:-1, -1]
    byc = wbxc[-1, -1]

    return wxc, bxc.ravel(), wyc.ravel(), byc

def comp_obj_vld(clf, lam, otherargs, vldx, vldy):
    m = vldx.shape[0]
    guess = clf.predict(vldx)
    guess = [1 if y_guess>=0 else -1 for y_guess in guess]
    errs = np.array(guess) - np.array(vldy)
    errs = errs.tolist()
    mse = np.linalg.norm(errs) ** 2 / m
    return mse

def comp_attack_vld(clf, wxc, bxc, wyc, byc, otherargs, vldx, vldy):
    n = vldx.shape[0]
    res = (clf.predict(vldx) - vldy)

    gradx = np.dot(vldx, wxc) + bxc
    grady = np.dot(vldx, wyc.T) + byc

    attackx = np.dot(res, gradx) / n
    attacky = np.dot(res, grady) / n

    return attackx, attacky






#############################################################################
####                               Read data                            #####
#############################################################################
### read for classification purpose
### last column as label (0 or 1), other are features
### for "spambase" we need an extra preprocess step
def read_dataset_file():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x1 = mnist.train.images  # Returns np.array, default [0,1.0], can be changed according to https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
    x2 = mnist.test.images
    x = np.concatenate((x1,x2),axis=0)
    print("x.shape:"+str(np.shape(x)))
    y1 = np.asarray(mnist.train.labels, dtype=np.int32)
    y2 = np.asarray(mnist.test.labels, dtype=np.int32)
    y = np.concatenate((y1,y2))
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


    inits = {'rand': rand_my,\
             'initialOutliers': initialOutliers_my}


    init = inits[args.initialization]

    initclf = linear_model.LinearRegression()
    initclf.fit(trainx, trainy)
    initlam = 0
    err = computeError(testx, testy, validx, validy,initclf)
    auc = computeTestAUC(testx, testy, initclf)

    print("----On clean set----")
    print("Validation MSE: {}".format(err[0]))
    print("Test AUC: {}".format(auc))
    print("Test MSE: {}".format(err[1]))
    clean_set_mse = err[1]
    clean_set_auc = auc

    # initialization step
    poisx, poisy, poisind = init(trainx, trainy, poisct, None)
    if args.flip == 1:
        poisy = [val * (-1) for val in poisy]
    trainx = np.matrix(trainx)

    clf = linear_model.LinearRegression()
    clf.fit(np.concatenate((trainx, poisx), axis=0), (trainy + poisy))
    err = computeError(testx, testy, validx, validy,clf)
    auc = computeTestAUC(testx, testy, clf)

    print("----After initialization (flipping?)----")
    print("Validation MSE: {}".format(err[0]))
    print("Test AUC: {}".format(auc))
    print("Test MSE: {}".format(err[1]))
    afterInitialization_mse = err[1]
    afterInitialization_auc = auc

    numsamples = poisct
    curpoisx = poisx[:numsamples, :]
    curpoisy = poisy[:numsamples]

    if args.optimizeAttack == 1:
        poisres, poisresy = poison_data(trainx, trainy, testx, testy, validx, validy,curpoisx, curpoisy)
    else:
        poisres, poisresy = poisx, poisy

    poisedx = np.concatenate((trainx, poisres), axis=0)
    poisedy = trainy + poisresy

    co = 1
    for pois in poisres:
        if np.isnan(np.sum(pois)):
            print("poisx "+str(co))
        co = co + 1
    if np.isnan(np.sum(poisresy)):
        print("poisresy")

    co = 1
    for pois in trainx:
        if np.isnan(np.sum(pois)):
            print("cleanx "+str(co))
        co = co + 1
    if np.isnan(np.sum(trainy)):
        print("cleany")



    clfp = linear_model.LinearRegression()
    clfp.fit(poisedx, poisedy)
    err = computeError(testx, testy, validx, validy, clfp)
    auc = computeTestAUC(testx, testy, clfp)
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
        main()
    else:
        x, y = read_dataset_file()
        store_CV_indices_inFile(x.shape[0], args.trainct, args.testct, args.validct, args.seed)

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))






