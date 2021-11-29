from sklearn.metrics import roc_auc_score
import argparse as ap
import os
from sklearn import preprocessing
import sys
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Dense, Flatten, MaxPooling2D
from numpy import *
from sklearn import *
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from numpy.random import seed
# from tensorflow import set_random_seed
import time


### please install the latest version of TensorFlow and uninstall the previous Keras before run this code



def read_params(args):
    parser = ap.ArgumentParser(description='Specify the probability')
    arg = parser.add_argument
    arg('-fn', '--fn', type=str, help='datasets')

    arg('-et', '--et', type=str, help='earlystopping')
    # arg('-ns', '--ns', type=str, help='number of select biomarkers')
    arg('-ts', '--ts', type=str, help='the ratio of test data')
    arg('-rs', '--rs', type=str, help='repeat times')

    return vars(parser.parse_args())


def read_files(file_name):


    # file_name='Karlsson_T2D'

    knownl = pd.read_csv(file_name + '_knownl.csv', index_col=0)
    knownp = pd.read_csv(file_name + '_knownp.csv', index_col=0)
    unknownl = pd.read_csv(file_name + '_unknownl.csv', index_col=0)
    unknownp = pd.read_csv(file_name + '_unknownp.csv', index_col=0)


    y = pd.read_csv("data/" + file_name+'_y.csv', index_col=0)
    le = preprocessing.LabelEncoder()
    y = np.array(y).ravel()
    y = le.fit_transform(y)
    return knownl, knownp, unknownl, unknownp, y


def transform_level(X):
    X = np.array(X)
    raw_dim = X.shape[1]
    img_size = math.ceil(raw_dim ** 0.5)
    new_dim = img_size ** 2

    add_blank = np.zeros((X.shape[0], new_dim - raw_dim))
    new_X = np.hstack((X, add_blank))
    new_X = new_X.reshape(X.shape[0], img_size, -1)
    print(new_X.shape)

    base_log = 4
    new_X = np.log(new_X + 1) / np.log(base_log)

    bins_break = [[0.0065536, np.max(new_X)],
                  [0.0016384, 0.0065536],
                  [0.0004096, 0.0016384],
                  [0.0001024, 0.0004096],
                  [0.0000256, 0.0001024],
                  [0.0000064, 0.0000256],
                  [0.0000016, 0.0000064],
                  [0.0000004, 0.0000016],
                  [0.0000001, 0.0000004],
                  [0, 0.0000001]]

    color_arry_num = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for windows in range(10):
        index = (new_X >= bins_break[windows][0]) & (new_X < bins_break[windows][1])
        new_X[index] = color_arry_num[windows]

    return new_X

def transform_post(X):
    X = np.array(X)
    raw_dim = X.shape[1]
    img_size = math.ceil(raw_dim ** 0.5)
    new_dim = img_size ** 2

    add_blank = np.zeros((X.shape[0], new_dim - raw_dim))
    new_X = np.hstack((X, add_blank))
    new_X = new_X.reshape(X.shape[0], img_size, -1)

    for img in new_X:
        for line in range(img.shape[0]):
            if line % 2 != 0:
                img[line] = img[line][::-1]
    print(new_X.shape)

    base_log = 4
    new_X = np.log(new_X + 1) / np.log(base_log)

    bins_break = [[0.0065536, np.max(new_X)],
                  [0.0016384, 0.0065536],
                  [0.0004096, 0.0016384],
                  [0.0001024, 0.0004096],
                  [0.0000256, 0.0001024],
                  [0.0000064, 0.0000256],
                  [0.0000016, 0.0000064],
                  [0.0000004, 0.0000016],
                  [0.0000001, 0.0000004],
                  [0, 0.0000001]]

    color_arry_num = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for windows in range(10):
        index = (new_X >= bins_break[windows][0]) & (new_X < bins_break[windows][1])
        new_X[index] = color_arry_num[windows]

    return new_X




par = read_params(sys.argv)

file_name = str(par['fn'])
ts = float(par['ts'])
rs = int(par['rs'])



et = str(par['et'])



if et=='t':
    earlystop = EarlyStopping(monitor='loss',
                              min_delta=1e-4,
                              patience=10,
                              verbose=1)
else:
    earlystop = EarlyStopping(monitor='val_loss',
                              min_delta=1e-4,
                              patience=10,
                              verbose=1)




def train_model(X_train, y_train, X_test, epochs, earlystop, et):
    K.clear_session()

    model = Sequential()
    model.add(Convolution2D(input_shape=(X_train.shape[1], X_train.shape[1], 1), kernel_size=(5, 5), filters=20,
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

    model.add(Convolution2D(kernel_size=(5, 5), filters=50, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    if et=='t':

        model.fit(X_train, y_train,
                  batch_size=X_train.shape[0],
                  epochs=epochs,
                  callbacks=[earlystop],
                  shuffle=True)

    else:
        model.fit(X_train, y_train,
                  batch_size=X_train.shape[0],
                  epochs=epochs,
                  validation_split=0.1,
                  callbacks=[earlystop],
                  shuffle=True)

    pred = model.predict(X_test)[:, 1]


    return pred




knownl, knownp, unknownl, unknownp, y=read_files(file_name)



knownl=transform_level(knownl)
knownp=transform_post(knownp)
unknownl=transform_level(unknownl)
unknownp=transform_post(unknownp)



knownl= np.expand_dims(knownl, axis=3)
knownp = np.expand_dims(knownp, axis=3)
unknownl= np.expand_dims(unknownl, axis=3)
unknownp = np.expand_dims(unknownp, axis=3)


start = time.time()

new_XL_train, new_XL_test, new_XP_train, \
new_XP_test, new_unXL_train, new_unXL_test, \
new_unXP_train, new_unXP_test, \
y_train, y_test = train_test_split(knownl, knownp, unknownl, unknownp, y, test_size=ts, random_state=4489)

y_train = to_categorical(y_train, num_classes=2)

print('++++++++++++++++++++++')
print(new_XL_train.shape)


ave_auc = []

ave_auc1 = []
ave_auc2 = []
ave_auc3 = []
ave_auc4 = []

ave_auc12 = []
ave_auc34 = []

repeat_seed = rs

for i in range(repeat_seed):
    K.clear_session()

    seed(i)
    tensorflow.random.set_seed(i)

    epochs = 500

    predL = train_model(new_XL_train, y_train, new_XL_test, epochs, earlystop,et)
    predP = train_model(new_XP_train, y_train, new_XP_test, epochs, earlystop,et)
    predunL = train_model(new_unXL_train, y_train, new_unXL_test, epochs, earlystop,et)
    predunP = train_model(new_unXP_train, y_train, new_unXP_test, epochs, earlystop,et)

    pred = (predL + predP + predunL + predunP) / 4

    pred1 = (predL)
    pred2 = (predP)
    pred3 = (predunL)
    pred4 = (predunP)

    pred12 = (predL + predP) / 2
    pred34 = (predunL + predunP) / 2

    auc = roc_auc_score(y_test, pred)

    auc1 = roc_auc_score(y_test, pred1)
    auc2 = roc_auc_score(y_test, pred2)
    auc3 = roc_auc_score(y_test, pred3)
    auc4 = roc_auc_score(y_test, pred4)

    auc12 = roc_auc_score(y_test, pred12)
    auc34 = roc_auc_score(y_test, pred34)

    # print('AUC for ' + str(i + 1) + ' rounds:  %.4f' % auc)
    ave_auc.append(auc)

    ave_auc1.append(auc1)
    ave_auc2.append(auc2)
    ave_auc3.append(auc3)
    ave_auc4.append(auc4)

    ave_auc12.append(auc12)
    ave_auc34.append(auc34)

# print(ave_auc)
# print('EPCNN Mean AUC :  %.4f' % mean(ave_auc))
mauc=mean(ave_auc)
# print(ave_auc1)
print('KnownL Mean AUC :  %.4f' % mean(ave_auc1))
mauc1=mean(ave_auc1)
# print(ave_auc2)
print('KnownP Mean AUC :  %.4f' % mean(ave_auc2))
mauc2=mean(ave_auc2)
# print(ave_auc3)
print('UnKnownL Mean AUC :  %.4f' % mean(ave_auc3))
mauc3=mean(ave_auc3)
# print(ave_auc4)
print('UnKnownP Mean AUC :  %.4f' % mean(ave_auc4))
mauc4=mean(ave_auc4)
# print(ave_auc12)
print('EnKnownL Mean AUC :  %.4f' % mean(ave_auc12))
mauc12=mean(ave_auc12)
# print(ave_auc34)
print('EnKnownP Mean AUC :  %.4f' % mean(ave_auc34))
mauc34=mean(ave_auc34)

end = time.time()
running_time = end - start

print(  'The best AUC of our EPCNN is: %.5f ' % max([mean(ave_auc),mean(ave_auc1),mean(ave_auc2),mean(ave_auc3), mean(ave_auc4),mean(ave_auc12),mean(ave_auc34)]) )

print('All Time cost : %.5f s' % running_time)


path = file_name + '_EPCNN_ev.txt'
if os.path.exists(path):
    os.remove(path)
file = open(path, 'a')

file.write('EPCNN Mean AUCs: ' + str(mauc) + "\n")

file.write('KnownL Mean AUCs: ' + str(mauc1) + "\n")
file.write('KnownP Mean AUCs: ' + str(mauc2) + "\n")
file.write('UnKnownL Mean AUCs: ' + str(mauc3) + "\n")
file.write('UnKnownP Mean AUCs: ' + str(mauc4) + "\n")
file.write('EnKnownL Mean AUCs: ' + str(mauc12) + "\n")
file.write('EnKnownP Mean AUCs: ' + str(mauc34) + "\n")
file.write('Time for ' + str(repeat_seed) + ' rounds running: ' + str(running_time) + "s")
