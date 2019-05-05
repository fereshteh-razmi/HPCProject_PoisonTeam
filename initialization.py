
from sklearn.neighbors import NearestNeighbors
from code_arg import setup_argparse
from code import *
import numpy as np

# ------------------------------------------------------------------------------\
### for classification, random selection
### Here we consider {-1,1} as labels
def rand_my(X_tr, Y_tr, count, poiser):
  poisinds = np.random.choice(X_tr.shape[0],count,replace=False)
  Y_pois = [Y_tr[i] for i in poisinds]
  return np.matrix(X_tr[poisinds]), Y_pois, poisinds

# ------------------------------------------------------------------------------\
### Select initial points that are outliers to the mean of training set
### then flip their labels
def initialOutliers_my(X_tr, Y_tr, pois_num, poiser):
    alpha_percentile = (1 - float(pois_num)/len(Y_tr)) * 100
    meanx = np.average(X_tr, axis=0)
    diff = np.linalg.norm((X_tr - meanx), axis=1)
    threshold = np.percentile(diff, alpha_percentile)
    outlier_ind = [ind for ind, val in enumerate(diff) if val >= threshold]
    Y_pois = [Y_tr[ind] for ind in outlier_ind]

    return np.matrix(X_tr[outlier_ind]), Y_pois ,outlier_ind

# ------------------------------------------------------------------------------\
### Select initial points that are outliers to the mean of training set
### then flip their labels
### Same as initialOutliers_my but first separate classes,then find outliers in each seprarately
def initialOutliers_sepClass_my(X_tr, Y_tr, pois_num, poiser):
    alpha_percentile = (1 - float(pois_num)/len(Y_tr)) * 100

    #z = [x for _, x in sorted(zip(Y, X))]
    train_pos_ind = [ind for ind, yn in enumerate(Y_tr) if yn == 1]
    train_pos = np.matrix(X_tr[train_pos_ind])
    meanx = np.average(train_pos, axis=0)
    diff = np.linalg.norm((train_pos - meanx), axis=1)
    threshold = np.percentile(diff, alpha_percentile)
    outlier_ind_pos = [ind for ind, val in zip(train_pos_ind,diff) if val >= threshold]

    train_neg_ind = [ind for ind, yn in enumerate(Y_tr) if yn == -1]
    train_neg = np.matrix(X_tr[train_neg_ind])
    meanx = np.average(train_neg, axis=0)
    diff = np.linalg.norm((train_neg - meanx), axis=1)
    threshold = np.percentile(diff, alpha_percentile)
    outlier_ind_neg = [ind for ind, val in zip(train_neg_ind,diff) if val >= threshold]

    outlier_ind = np.concatenate((outlier_ind_pos,outlier_ind_neg),axis=0)
    outlier_ind.sort()
    Y_pois = [Y_tr[ind] for ind in outlier_ind]
    return np.matrix(X_tr[outlier_ind]), Y_pois ,outlier_ind


# ------------------------------------------------------------------------------\
### Select initial points that are KNN outliers of training set
### then flip their labels
def initialOutliers_KNN_my(X_tr, Y_tr, pois_num, poiser):
    alpha_percentile = (1 - float(pois_num)/len(Y_tr)) * 100
    # Number of NN , since NearestNeighbors() returns the point itself as the NN,
    # so for actual NN we need to put k = 2
    k = 2

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X_tr)
    distances, indices = nbrs.kneighbors(X_tr)
    diff = []
    # for i in range(len(X_tr)):
    #     curr_x = X_tr[i].ravel()
    #     curr_NNs_dist = [np.linalg.norm(curr_x - X_tr[ind].ravel()) for ind in range(len(X_tr))]
    #     curr_NNs_dist.sort()
    #     curr_NNs_dist = curr_NNs_dist[1:] # remove distance 0 from curr_x to itself
    #     diff.append(curr_NNs_dist[k])

    diff = distances[:,1]
    threshold = np.percentile(diff, alpha_percentile)
    outlier_ind = [ind for ind, val in enumerate(diff) if val >= threshold]
    Y_pois = [Y_tr[ind] for ind in outlier_ind]

    return np.matrix(X_tr[outlier_ind]), Y_pois ,outlier_ind


# -------------------------------------------------------------------------------
def cookfDist_my(x, y, count, poiser):
    allpoisy = []
    clf, _ = poiser.learn_model(x, y, None)
    preds = [clf.predict(x[i].reshape(1, -1)) for i in range(x.shape[0])]
    errs = [(y[i] - preds[i]) ** 2 for i in range(x.shape[0])]
    mean = np.ravel(x.mean(axis=0))  # .reshape(1,-1)
    corr = np.dot(x.T, x) + 0.01 * np.eye(x.shape[1])
    invmat = np.linalg.pinv(corr)
    hmat = x * invmat * np.transpose(x)

    allcooks = [hmat[i, i] * errs[i] / (1 - hmat[i, i]) ** 2 for i in range(x.shape[0])]
    #indices = np.argsort(allcooks)
    indices = sorted(range(len(allcooks)), key=lambda xn: allcooks[xn] , reverse=True)

    poisinds = indices[0:count]
    X_pois = x[poisinds]
    allpoisy = [y[ind] for ind in poisinds]

    return x[poisinds], allpoisy, poisinds

# ------------------------------------------------------------------------------\
### Implementation of Robustness of Neural Ensembles Against Targeted and Random Adversarial Learning
### calculates Mahalanobis distance(D) + Box M Test statistics(C): the smaller they are ,the more representative sample is
### flip representative sample labels
def representative_my(X_tr, Y_tr, count, poiser):

    X_mean = np.average(X_tr, axis=0).ravel()
    n1 = len(X_tr)
    X_cov = np.cov(X_tr.T)
    DC = []
    all_poisinds = []
    for i in range(1000):
        poisinds = np.random.choice(X_tr.shape[0], count, replace=False)
        #if i == 0:
        all_poisinds.append(poisinds)
        #else:
        #    all_poisinds = np.append(all_poisinds, poisinds, axis=1)
        X_sample = np.matrix(X_tr[poisinds])
        sample_mean = np.average(X_sample, axis=0)
        n2 = len(X_sample)
        #omitted: 1/(n2-1) * ...
        sample_covar = np.cov(X_sample.T)
        pooled_cov  = ( (n1-1)*X_cov + (n2-1)*sample_covar ) / ( n1 + n2 - 2 )
        D = (X_mean - sample_mean)*np.linalg.pinv(pooled_cov)*(X_mean - sample_mean).T
        D = np.linalg.norm(D)

        # Boxs M Test
        C = (n1+n2-2)*np.log(np.linalg.norm(pooled_cov)) - (n1-1)*np.log(np.linalg.norm(X_cov)) - (n2-1)*np.log(np.linalg.norm(sample_covar))
        DC.append(D * C)

    indices = sorted(range(len(DC)), key=lambda xi: DC[xi])
    sample_num = np.random.choice(indices[1:50], 1)
    sample_num = np.asscalar(sample_num)
    selected_pois_inds = all_poisinds[sample_num][:]

    Y_pois = [Y_tr[i] for i in selected_pois_inds]
    return np.matrix(X_tr[selected_pois_inds]), Y_pois, selected_pois_inds
