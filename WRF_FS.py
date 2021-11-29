import argparse as ap
import pandas as pd
import numpy as np
import sys
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import time
import os
from scipy import stats
from numpy.random import seed
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


def read_params(args):
    parser = ap.ArgumentParser(description='Specify the probability')
    arg = parser.add_argument
    arg('-fn', '--fn', type=str, help='datasets')

    arg('-tp', '--tp', type=str, help='number of select features')

    arg('-rs', '--rs', type=str, help='repeat times')

    return vars(parser.parse_args())


def read_files(file_name):


    # file_name='Karlsson_T2D'

    known = pd.read_csv("data/" +file_name+'_known.csv', index_col=0)
    unknown = pd.read_csv("data/" +file_name+'_unknown.csv', index_col=0)

    y = pd.read_csv("data/" +file_name+'_y.csv', index_col=0)
    le = preprocessing.LabelEncoder()

    y=np.array(y).ravel()
    y = le.fit_transform(y)
    return known, unknown, y


def WRF_sff(known, unknown, y):


    known_X = np.array(known)
    unknown_X = np.array(unknown)

    paramsampler = {'max_features': stats.uniform(0, 1.0),
                    'max_depth': stats.randint(1, 10), "n_estimators": stats.randint(100, 2000)}

    clfx = RandomizedSearchCV(
        RandomForestClassifier(random_state=4487),
        param_distributions=paramsampler,
        cv=5, n_jobs=-1, random_state=4489)

    clfx.fit(known_X, y)

    clfz = RandomizedSearchCV(
        RandomForestClassifier(random_state=4487),
        param_distributions=paramsampler,
        cv=5, n_jobs=-1, random_state=4489)

    clfz.fit(unknown_X, y)

    known_oob = clfx.best_score_
    unknown_oob = clfz.best_score_

    known_weight = known_oob / (known_oob + unknown_oob)

    unknown_weight = 1 - known_weight

    combinedX = np.hstack((known_weight * known_X, unknown_weight * unknown_X))

    paramsampler = {'max_features': stats.uniform(0, 1.0),
                    'max_depth': stats.randint(1, 10), "n_estimators": stats.randint(100, 2000)}
    clfc = RandomizedSearchCV(
        RandomForestClassifier(random_state=4487),
        param_distributions=paramsampler,
        cv=5, n_jobs=-1, random_state=4489)

    clfc.fit(combinedX, y)

    rf = clfc.best_estimator_

    model = SelectFromModel(rf, prefit=True)

    slf = model.get_support()
    slf = list(slf)
    slf=[int(i) for i in slf]
    return known_weight, unknown_weight, slf





par = read_params(sys.argv)

file_name = str(par['fn'])
tp = int(par['tp'])
rs = int(par['rs'])



known, unknown,y=read_files(file_name)

raw_fea = known.columns.values.tolist()
un_fea = unknown.columns.values.tolist()
feature_lab = raw_fea + un_fea
print('all_feature_length: ' + str(len(feature_lab)))



all_score=[]
for i in range(rs):
    seed(i)
    top30_rawfeature=WRF_sff(known, unknown,y)
    all_score.append(top30_rawfeature[2])

all_score=np.array(all_score).reshape(rs,-1)

ave_importances=np.mean(all_score, axis=0)

sfn=[i for i in ave_importances if i!=0]

indices = np.argsort(ave_importances)[::-1]


weighted_features=[]
for f in range(tp):
    weighted_features.append(feature_lab[indices[f]])

print('Selected features: ' + str(len(sfn)))
print('Top '+str(tp)+' features: ' + "\n"+ str(weighted_features))

path = file_name + '_WRF_fs.txt'
if os.path.exists(path):
    os.remove(path)
file = open(path, 'a')

file.write('Selected features: ' + str(len(sfn))+ "\n")
file.write('Top '+str(tp)+' features: ' + "\n"+ str(weighted_features))