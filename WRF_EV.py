import argparse as ap
import pandas as pd
import numpy as np
import sys
from sklearn import preprocessing


def read_params(args):
    parser = ap.ArgumentParser(description='Specify the probability')
    arg = parser.add_argument
    arg('-fn', '--fn', type=str, help='datasets')
    # arg('-ns', '--ns', type=str, help='number of select biomarkers')
    arg('-ts', '--ts', type=str, help='the ratio of test data')
    arg('-rs', '--rs', type=str, help='repeat times')

    return vars(parser.parse_args())


def read_files(file_name):


    # file_name='Karlsson_T2D'

    known = pd.read_csv("data/" + file_name+'_known.csv', index_col=0)
    unknown = pd.read_csv("data/" + file_name+'_unknown.csv', index_col=0)

    y = pd.read_csv("data/" + file_name+'_y.csv', index_col=0)
    le = preprocessing.LabelEncoder()
    y = np.array(y).ravel()
    y = le.fit_transform(y)
    return known, unknown, y



def WRF_eva(known, unknown,y, ts, rs, file_name ):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    from sklearn.metrics import roc_auc_score
    import time
    import os
    from scipy import stats
    from numpy.random import seed
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier

    start = time.time()

    known_X = np.array(known)
    unknown_X = np.array(unknown)

    knownX_train, knownX_test, unknownX_train, unknownX_test, y_train, y_test = train_test_split(
        known_X, unknown_X, y, test_size=ts, random_state=4489)

    ave_auc = []
    repeat_seed = rs

    path = file_name+'_WRF_ev.txt'
    if os.path.exists(path):
        os.remove(path)
    file = open(path, 'a')



    feature_imp = []

    kweight = []

    for i in range(repeat_seed):

        print('Round ' + str(i + 1))

        seed(i)
        y_predpro = []


        paramsampler = {'max_features': stats.uniform(0, 1.0),
                        'max_depth': stats.randint(1, 10), "n_estimators": stats.randint(100, 2000)}

        clf = RandomizedSearchCV(
            RandomForestClassifier(oob_score=True),
            param_distributions=paramsampler,
            cv=5, n_jobs=-1)

        clf.fit(knownX_train, y_train)

        clfz = RandomizedSearchCV(
            RandomForestClassifier(oob_score=True),
            param_distributions=paramsampler,
            cv=5, n_jobs=-1)

        clfz.fit(unknownX_train, y_train)

        known_oob = clf.best_score_
        unknown_oob = clfz.best_score_

        known_weight = known_oob / (known_oob + unknown_oob)

        unknown_weight = 1 - known_weight

        kweight.append(known_weight)

        print('known weight:  %.4f' % known_weight, 'unknown weight: %.4f' % unknown_weight)






        combinedX_train = np.hstack((known_weight * knownX_train, unknown_weight * unknownX_train))

        combinedX_test = np.hstack((known_weight * knownX_test, unknown_weight * unknownX_test))

        clfc = RandomizedSearchCV(
            RandomForestClassifier(),
            param_distributions=paramsampler,
            cv=5, n_jobs=-1)

        clfc.fit(combinedX_train, y_train)

        importances = clfc.best_estimator_.feature_importances_

        feature_imp.append(importances)

        pred_prob = clfc.predict_proba(combinedX_test)[:, 1]

        y_predpro.extend(pred_prob)

        auc = roc_auc_score(y_test, pred_prob)

        ave_auc.append(auc)

        print('AUC :  %.4f' % auc)

    print('Mean AUC :  %.4f' % np.mean(ave_auc))
    print('Ave Known Weight :  %.4f' % np.mean(kweight))
    print('Ave unKnown Weight :  %.4f' % (1 - np.mean(kweight)))

    end = time.time()
    running_time = end - start
    print('Time cost : %.5f s' % running_time)


    meanauc=np.mean(ave_auc)
    mkweight=np.mean(kweight)
    mukweight=1 - np.mean(kweight)
    print('====================')
    file.write('Mean AUCs: ' + str(meanauc) + "\n")
    file.write('Mean Known weights: ' + str(mkweight) + "\n")
    file.write('Mean Unknown weights: ' + str(mukweight) + "\n")

    file.write('Time for '+str(repeat_seed)+' rounds running: ' + str(running_time) + "s")


par = read_params(sys.argv)

file_name = str(par['fn'])
ts = float(par['ts'])
rs = int(par['rs'])

known, unknown,y=read_files(file_name)
WRF_eva(known, unknown,y, ts, rs, file_name )

