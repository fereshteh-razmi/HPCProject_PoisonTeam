from __future__ import division, print_function

import code
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


def __init__(self, x, y, testx, testy, validx, validy, \
            eta, beta, sigma, eps, mproc, objective):

    self.trnx = x
    self.trny = y
    self.tstx = testx
    self.tsty = testy
    self.vldx = validx
    self.vldy = validy

    self.samplenum = x.shape[0]
    self.feanum = x.shape[1]


    self.mp = mproc # use multiprocessing?

    self.eta = eta
    self.beta = beta
    self.sigma = sigma
    self.eps = eps

    self.initclf,self.initlam = None,None


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

    best_obj = 0
    count = 0
    print("before pooling")
    if args.multiproc:
        import multiprocessing as mp
        print("after importing")
        workerpool = 1#mp.Pool()#max(1, mp.cpu_count() // 2 - 1)
        print("pooooool number: "+str(mp.cpu_count()))
        sys.stdout.flush()
    else:
        workerpool = 0#None

    print("after pooling")
    sys.stdout.flush()
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
    while True:
        start = time.time()
        count += 1
        print("count =" + str(count)) #######
        sys.stdout.flush()
        new_poisx = np.matrix(np.zeros(poisx.shape))
        new_poisy = [None for a in poisy]
        x_cur = np.concatenate((trnx, poisx), axis=0)
        y_cur = trny + poisy

        samplenum = trnx.shape[0]
        feanum = trnx.shape[1]
        clf, lam = learn_model(x_cur, y_cur, None)
        pois_params = [(poisx[i], poisy[i], eq7lhs, mu, clf, lam, samplenum, feanum, trnx, trny, vldx, vldy) \
                       for i in range(poisct)]
        outofboundsct = 0
        pool = None
        if workerpool == 1:  # multiprocessing
            pool = mp.Pool()
            # print("here")
            # sys.stdout.flush()
            #pool.map(unwrap_self_f, zip([self]*len(names), names))
            for i, cur_pois_res in enumerate(
                    #workerpool.map(unwrap_self_f, zip([self]*len(pois_params),pois_params))):#
                     pool(poison_data_subroutine,pois_params)):
                new_poisx[i] = cur_pois_res[0]
                new_poisy[i] = cur_pois_res[1]
                outofboundsct += cur_pois_res[2]

        else:
            # print("there")
            for i in range(poisct):
                cur_pois_res = poison_data_subroutine(pois_params[i])

                new_poisx[i] = cur_pois_res[0]
                new_poisy[i] = cur_pois_res[1]
                outofboundsct += cur_pois_res[2]


        it_res = iter_progress(trnx, trny, poisx, poisy, new_poisx, new_poisy)


        if (it_res[0] <= it_res[1]):
            args.eta *= 0.75
            poisx, poisy = new_poisx, new_poisy ###### added
        else:
            best_count = count #######
            best_res = it_res[0] #######
            poisx = new_poisx
            poisy = new_poisy


        if (it_res[0] > best_obj):
            best_poisx, best_poisy, best_obj = poisx, poisy, it_res[0] ### changed it_res[1] to it_res[0]

        it_diff = abs(it_res[0] - it_res[1])

        # stopping conditions
        if (count >= 5 and (it_diff <= args.eps or count > 5)):
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

    poisxelem, poisyelem, eq7lhs, mu, clf, lam , samplenum, feanum, trnx, trny, vldx, vldy = in_tuple
    m = compute_m(clf, poisxelem, poisyelem, samplenum, feanum)
    feanum = poisxelem.shape[1]
    # compute partials

    wxc, bxc, wyc, byc = compute_wb_zc(eq7lhs, mu, clf.coef_, m, \
                                            samplenum, poisxelem)


    otherargs = None

    attack, attacky = comp_attack_vld(clf, wxc, bxc, wyc, byc, otherargs, vldx, vldy)

    # keep track of how many points are pushed out of bounds
    #### we don't use it for Classification but be cautious if we did, had to change 0 to -1
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

    poisxelem, poisyelem, _ = lineSearch(trnx, trny, vldx, vldy,poisxelem, poisyelem, \
                                              attack, attacky)
    poisxelem = poisxelem.reshape((1, feanum))

    return poisxelem, poisyelem, outofbounds


def lineSearch(trnx, trny, vldx, vldy, poisxelem, poisyelem, attack, attacky):
    k = 0
    x0 = np.copy(trnx)
    y0 = trny[:]

    curx = np.append(x0, poisxelem, axis=0)
    cury = y0[:]  # why not?
    cury.append(poisyelem)

    clf, lam = learn_model(curx, cury, None)
    clf1, lam1 = clf, lam

    lastpoisxelem = poisxelem
    curpoisxelem = poisxelem

    lastyc = poisyelem
    curyc = poisyelem
    otherargs = None

    w_1 = comp_obj_vld(clf, lam, otherargs, vldx, vldy)
    count = 0
    eta = args.eta

    while True:
        if (count > 0):
            eta = args.beta * eta
        count += 1
        curpoisxelem = curpoisxelem + eta * attack
        curpoisxelem = np.clip(curpoisxelem, 0, 1)
        curx[-1] = curpoisxelem
        clf1, lam1 = learn_model(curx, cury, clf1)
        w_2 = comp_obj_vld(clf1, lam1, otherargs, vldx, vldy)

        if (count >= 100):# or abs(w_1 - w_2) < 1e-8):  # convergence
            print('100')
            sys.stdout.flush()
            break
        if (w_2 - w_1 < 0):  # bad progress
            curpoisxelem = lastpoisxelem
            curyc = lastyc
            print(str(count))
            sys.stdout.flush()
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
        ######these lines I added, they are for logistic regression
        if guess < 0 :
            guess = -1
        else:
            guess = 1
        err = guess - trueval
        #print(str(i) + ":" + str(guess) + "," + str(trueval))

        v_toterr += err ** 2  # MSE
        v_rsqnum += (guess - vmean) ** 2  # R^2 num and denom
        v_rsqdenom += (trueval - vmean) ** 2

    for i, trueval in enumerate(tsty):
        guess = pred[i]
        ######these lines I added, they are for logistic regression
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
        clf = linear_model.LinearRegression()


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


