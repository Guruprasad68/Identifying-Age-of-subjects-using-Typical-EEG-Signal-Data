#%% import packages
import numpy as np, pandas as pd, matplotlib.pyplot as plt, os, time, pickle, librosa as lr
from scipy.stats import ttest_ind
from sklearn.metrics import mutual_info_score
# import scikit_posthocs as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import random

#%% code
path = '___'
only32 = True
if only32 == True:
    with open(path + 'data/train32.pkl', 'rb') as f:
        L = pickle.load(f)
else:
    L = os.listdir(path+'data/data_eeg_age_v1/data2kaggle/train/')
data_path = path+'data/data_eeg_age_v1/data2kaggle/'
dataset = 'train/'
# ns = int(235000 / 2)
bs = 23; bstr = 17; bste = 6 


skip_dc = True
if skip_dc == False:

    if os.path.isfile('___/Trdata32_svm.pkl') == True:
        with open('___/Trdata32_svm.pkl', 'rb') as f:
            D = pickle.load(f)
        X_train = D['x_train']
        X_test = D['x_test']
        Ytr = D['y_train']
        Yte = D['y_test']
    else:
        r = random.sample(range(bs), bs)
        # uik
        for items in L[0:200]:
            df = pd.read_csv(data_path + dataset+ items, skiprows = [0, 1])
            h = pd.read_csv(data_path + dataset+ items, index_col=0, nrows=0).columns.tolist()[0]
            h = int(h.split('= ')[1])
            h_ = np.argmax([10<=h<=30, 31<=h<=40, 41<=h<=50, 51<=h<=60, 61<=h<=90])
            vals_tr = []
            vals_te = []
            ytr = []
            yte = []

            ktest = 0
            ktrain = 0
            ktr = 0
            kte = 0
            # xte = np.zeros((1,2))
            for i in range(0, df.to_numpy().shape[1] - 4):
                xtr = []
                xte = []
                X = df.to_numpy()[:, i][0:230000]
                ns = int(X.shape[0] / bs)
                for ih in r:
                    # print(i, ih, len(r), L.index(items))
                    temp = lr.feature.rms(X[ih*ns : (ih+1)*ns], frame_length = 512, hop_length = 256)
                    if len(xtr) <= 17:
                        xtr.append(temp.squeeze())
                        if ktr == 0:
                            ytr.append(h_)
                    else:
                        xte.append(temp.squeeze())               
                        if kte == 0:
                            yte.append(h_)
                if i == 0:
                    Xtr = np.array(xtr)
                    Xte = np.array(xte)
                else:
                    Xtr = np.concatenate((Xtr, np.array(xtr)), axis = 1)
                    Xte = np.concatenate((Xte, np.array(xte)), axis = 1)
                    # uik                
                ktr = 1
                kte = 1
                # uik
            # uik
            if L.index(items) == 0:
                X_train = Xtr
                X_test = Xte
                Y_train = ytr
                Y_test = yte
            else:
                X_train = np.concatenate((X_train, Xtr), axis = 0)
                X_test = np.concatenate((X_test, Xte), axis = 0)
                Y_train.extend(ytr)
                Y_test.extend(yte)
                # uik
            print('[info 3]', L.index(items), len(L), X_train.shape, len(Y_train), X_test.shape, len(Y_test), len(ytr), len(yte))
else:
    #%% load full data
    with open('___/Trdata32_svm.pkl', 'rb') as f:
        data32 = pickle.load(f)
    # full subjects data
    X_train = data32['x_train']; X_test= data32['x_test']
    Y_train = data32['y_train']; Y_test = data32['y_test']
# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test) 


# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

#%% run sklearn classifiers

Y_P_tr = {}
Y_P_te = {}
tr_accs = {}
te_accs = {}
Y_T_tr = {}
Y_T_te = {}
clfs = {}

from sklearn.svm import SVC
print('running svm')

start = time.time()
clf = SVC(C = 1, kernel = 'linear', probability = True, verbose = True)
clf.fit(X_train, Y_train)

y_pred_train = clf.predict(X_train)
tracc = accuracy_score(Y_train, y_pred_train)
tr_accs['trainSVM'] = tracc

y_pred_test = clf.predict(X_test)
teacc = accuracy_score(Y_test, y_pred_test)
te_accs['testSVM'] = teacc
stop = time.time()

Y_P_tr['SVM'] = y_pred_train
Y_P_te['SVM'] = y_pred_test
Y_T_tr['SVM'] = Y_train
Y_T_te['SVM'] = Y_test
clfs['SVM'] = clf
print('SVM Done; time elapsed:', (stop-start)/60, '\nTracc:', tracc, 'Teacc', teacc)

start = time.time()
print('LDA')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()
clf.fit(X_train, Y_train)
y_pred_train = clf.predict(X_train)
tracc = accuracy_score(Y_train, y_pred_train)

y_pred_test = clf.predict(X_test)
teacc = accuracy_score(Y_test, y_pred_test)
stop = time.time()
print('LDA Done; time elapsed:', (stop-start)/60, '\nTracc:', tracc, 'Teacc', teacc)
tr_accs['trainLDA'] = tracc
te_accs['testLDA'] = teacc
Y_P_tr['LDA'] = y_pred_train
Y_P_te['LDA'] = y_pred_test
Y_T_tr['LDA'] = Y_train
Y_T_te['LDA'] = Y_test
clfs['LDA'] = clf
# QDA
start = time.time()
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
clf = QuadraticDiscriminantAnalysis(reg_param = 1)
clf.fit(X_train, Y_train)
y_pred_train= clf.predict(X_train)
y_pred_test = clf.predict(X_test)
tracc = accuracy_score(Y_train, y_pred_train)
teacc = accuracy_score(Y_test, y_pred_test)
tr_accs['trainQDA'] = tracc
te_accs['testQDA'] = teacc
# Y_P['svm'] = y_pred
stop = time.time()
Y_P_tr['QDA'] = y_pred_train
Y_P_te['QDA'] = y_pred_test
Y_T_tr['QDA'] = Y_train
Y_T_te['QDA'] = Y_test
clfs['QDA'] = clf
print((stop - start)/60)
print('QDA Done; time elapsed:', (stop-start)/60, '\nTracc:', tracc, 'Teacc', teacc)

#% log reg
start = time.time()
print('LR')
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0)#.fit(x_train, y_train)
clf.fit(X_train, Y_train)
y_pred_train= clf.predict(X_train)
y_pred_test = clf.predict(X_test)
tracc = accuracy_score(Y_train, y_pred_train)
teacc = accuracy_score(Y_test, y_pred_test)
tr_accs['trainLR'] = tracc
te_accs['testLR'] = teacc
s = clf.score(X_test, Y_test)
Y_P_tr['LR'] = y_pred_train
Y_P_te['LR'] = y_pred_test
Y_T_tr['LR'] = Y_train
Y_T_te['LR'] = Y_test
clfs['LR'] = clf
stop = time.time()
print((stop - start)/60)
print('LR Done; time elapsed:', (stop-start)/60, '\nTracc:', tracc, 'Teacc', teacc)

#% decision tree
print('DT')
start = time.time()
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, Y_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
tracc = accuracy_score(Y_train, y_pred_train)
teacc = accuracy_score(Y_test, y_pred_test)
tr_accs['trainDT'] = tracc
te_accs['testDT'] = teacc
Y_P_tr['DT'] = y_pred_train
Y_P_te['DT'] = y_pred_test
Y_T_tr['DT'] = Y_train
Y_T_te['DT'] = Y_test
clfs['DT'] = clf
stop = time.time()
print((stop - start)/60)
print('GNB Done; time elapsed:', (stop-start)/60, '\nTracc:', tracc, 'Teacc', teacc)

#% GNB
start = time.time()
print('GNB')
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, Y_train)
y_pred_train= clf.predict(X_train)
y_pred_test = clf.predict(X_test)
tracc = accuracy_score(Y_train, y_pred_train)
teacc = accuracy_score(Y_test, y_pred_test)
tr_accs['trainGNB'] = tracc
te_accs['testGNB'] = teacc
Y_P_tr['GNB'] = y_pred_train
Y_P_te['GNB'] = y_pred_test
Y_T_tr['GNB'] = Y_train
Y_T_te['GNB'] = Y_test
clfs['DT'] = clf
stop = time.time()
print((stop - start)/60)
print('DT Done; time elapsed:', (stop-start)/60, '\nTracc:', tracc, 'Teacc', teacc)

#% KNN
start = time.time()
print('KNN')
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=15)
clf.fit(X_train, Y_train)
y_pred_train= clf.predict(X_train)
y_pred_test = clf.predict(X_test)
tracc = accuracy_score(Y_train, y_pred_train)
teacc = accuracy_score(Y_test, y_pred_test)
tr_accs['trainKNN'] = tracc
te_accs['testKNN'] = teacc
Y_P_tr['KNN'] = y_pred_train
Y_P_te['KNN'] = y_pred_test
Y_T_tr['KNN'] = Y_train
Y_T_te['KNN'] = Y_test
clfs['KNN'] = clf
stop = time.time()
print((stop - start)/60)
print('KNN Done; time elapsed:', (stop-start)/60, '\nTracc:', tracc, 'Teacc', teacc)

#% RF
start = time.time()
print('RF')
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=16, random_state=0)
clf.fit(X_train, Y_train)
y_pred_train= clf.predict(X_train)
y_pred_test = clf.predict(X_test)
tracc = accuracy_score(Y_train, y_pred_train)
teacc = accuracy_score(Y_test, y_pred_test)
tr_accs['trainRF'] = tracc
te_accs['testRF'] = teacc
Y_P_tr['RF'] = y_pred_train
Y_P_te['RF'] = y_pred_test
Y_T_tr['RF'] = Y_train
Y_T_te['RF'] = Y_test
clfs['RF'] = clf
stop = time.time()
print((stop - start)/60)
print('KNN Done; time elapsed:', (stop-start)/60, '\nTracc:', tracc, 'Teacc', teacc)

#% XGB Tree
start = time.time()
print('XGB')
from xgboost import XGBClassifier
clf= XGBClassifier()
clf.fit(X_train, Y_train)
y_pred_train= clf.predict(X_train)
y_pred_test = clf.predict(X_test)
tracc = accuracy_score(Y_train, y_pred_train)
teacc = accuracy_score(Y_test, y_pred_test)
tr_accs['trainXGB'] = tracc
te_accs['testXGB'] = teacc
Y_P_tr['XGB'] = y_pred_train
Y_P_te['XGB'] = y_pred_test
Y_T_tr['XGB'] = Y_train
Y_T_te['XGB'] = Y_test
clfs['XGB'] = clf
stop = time.time()
print((stop - start)/60)
print('XGB Done; time elapsed:', (stop-start)/60, '\nTracc:', tracc, 'Teacc', teacc)

#%%
import pickle
resat6 = {'X_train': X_train, 'Y_train': Y_train, 'X_test': X_test, 'Y_test': Y_test, 'clfs': clfs, 'tr_accs': tr_accs, 'te_accs': te_accs, 'Y_P_tr': Y_P_tr, 'Y_P_te': Y_P_te, 'Y_T_tr': Y_T_tr, 'Y_T_te': Y_T_te} # svm in resat6.pkl here to be neglected
with open('___/resat6_20.pkl', 'wb') as f:
    pickle.dump(resat6, f)