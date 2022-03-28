#%% import packages
import numpy as np, pandas as pd, matplotlib.pyplot as plt, os, time, pickle, librosa as lr
from scipy.stats import ttest_ind
from sklearn.metrics import mutual_info_score
# import scikit_posthocs as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import random
np.random.seed(1)
#%% code
path = 'C:/UWMad/Subjects/F21/ECE539/course_project/'
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
# print('FE started')
# start = time.time()
if os.path.isfile('C:/UWMad/Subjects/F21/ECE539/course_project/data/Trdata32_svm.pkl') == False:
    with open('C:/UWMad/Subjects/F21/ECE539/course_project/data/Trdata32_svm.pkl', 'rb') as f:
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

# =============================================================================
# scaler = StandardScaler()
# 
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# =============================================================================

data32 = {'x_train': X_train, 'x_test': X_test, 'y_train': Y_train, 'y_test': Y_test}
with open('C:/UWMad/Subjects/F21/ECE539/course_project/data/Trdata32_svm.pkl', 'wb') as f:
    pickle.dump(data32, f)

# # =============================================================================
# # with open('C:/UWMad/Subjects/F21/ECE539/course_project/data/Trdata32_svm.pkl', 'rb') as f:
# #     data32 = pickle.load(f)
# # 
# # X_train = data32['x_train']; X_test= data32['x_test']
# # Ytr = data32['y_train']; Yte = data32['y_test']
# # =============================================================================
# 
# scaler = StandardScaler()
# 
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# uik
#%%
# def classify_data(clf, X_train, Y_train, X_test, Y_test)
Y_P = {}
tr_accs = {}
te_accs = {}
Y_T = {}

#%% load full data
with open('C:/UWMad/Subjects/F21/ECE539/course_project/data/Trdata32_svm.pkl', 'rb') as f:
    data32 = pickle.load(f)
# full subjects data
X_train = data32['x_train']; X_test= data32['x_test']
Y_train = data32['y_train']; Y_test = data32['y_test']
# =============================================================================
# scaler = StandardScaler()
# 
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# =============================================================================
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
clf = SVC(C = 0.07, kernel = 'linear', probability = True, verbose = True)
clf.fit(X_train, Y_train)
# X_test = 
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

# =============================================================================
# from sklearn.svm import SVC
# print('running svm')
# start = time.time()
# clf = SVC(C = 0.07, kernel = 'linear', probability = True, verbose = True)
# clf.fit(X_train, Y_train)
# # X_test = 
# y_pred_train = clf.predict(X_train)
# tracc = accuracy_score(Y_train, y_pred_train)
# tr_accs['trainSVM'] = tracc
# 
# y_pred_test = clf.predict(X_test)
# teacc = accuracy_score(Y_test, y_pred_test)
# te_accs['testSVM'] = teacc
# stop = time.time()
# print('SVM Done; time elapsed:', (stop-start)/60, '\nTracc:', tracc, 'Teacc', teacc)
# =============================================================================
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

# =============================================================================
# resat6 = {'clmtraccs': tr_accs, 'clmteaccs': te_accs} # svm in resat6.pkl here to be neglected
# with open('C:/UWMad/Subjects/F21/ECE539/course_project/data/resat6.pkl', 'wb') as f:
#     pickle.dump(resat6, f)
# =============================================================================

# =============================================================================
# 
# =============================================================================
uik
#%%
resat6 = {'svmclf': clf, 'y_pred_train': y_pred_train, 'y_pred_test': y_pred_test, 'tracc': tracc, 'teacc': teacc}
with open('C:/UWMad/Subjects/F21/ECE539/course_project/data/resat6.pkl', 'wb') as f:
    pickle.dump(resat6, f)
#     
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clfl = LinearDiscriminantAnalysis()
clfl.fit(X_train, Y_train)
y_pred_train = clf.predict(X_train)
tracc = accuracy_score(Y_train, y_pred_train)

y_pred_test = clfl.predict(X_test)
teacc = accuracy_score(Y_test, y_pred_test)


#%%
import pickle
resat6 = {'X_train': X_train, 'Y_train': Y_train, 'X_test': X_test, 'Y_test': Y_test, 'clfs': clfs, 'tr_accs': tr_accs, 'te_accs': te_accs, 'Y_P_tr': Y_P_tr, 'Y_P_te': Y_P_te, 'Y_T_tr': Y_T_tr, 'Y_T_te': Y_T_te} # svm in resat6.pkl here to be neglected
with open('C:/UWMad/Subjects/F21/ECE539/course_project/data/resat6_20.pkl', 'wb') as f:
    pickle.dump(resat6, f)
# C:\UWMad\Subjects\F21\ECE539\course_project\data
#%%
print(clfs['RF'].feature_importances_)
plt.bar(range(len(clfs['RF'].feature_importances_)), clfs['RF'].feature_importances_)
plt.show()
#%%
mdl = clfs['XGB']
# from xgboost import plot_importance
# plot_importance(clf)
f_imp = mdl.feature_importances_
feat_num = -50
# =============================================================================
# plt.bar(range(len(f_imp)), f_imp)
# plt.stem(390 - 1, f_imp[390-1])
# plt.show()
# =============================================================================
idxs = np.argsort(f_imp)
f_names = range(1, 1281)
features = X_train
plt.title('Feature Importances')
plt.barh(range(len(idxs[feat_num:])), f_imp[idxs[feat_num:]], color='b', align='center')
plt.yticks(range(len(idxs[feat_num:])), [f_names[i] for i in idxs[feat_num:]])
plt.xlabel('Relative Importance')
plt.show()


#%% this 1
# resat6 = {'X_train': X_train, 'Y_train': Y_train, 'X_test': X_test, 'Y_test': Y_test, 'clfs': clfs, 'tr_accs': tr_accs, 'te_accs': te_accs, 'Y_P_tr': Y_P_tr, 'Y_P_te': Y_P_te, 'Y_T_tr': Y_T_tr, 'Y_T_te': Y_T_te} # svm in resat6.pkl here to be neglected
with open('C:/UWMad/Subjects/F21/ECE539/course_project/data/resat6_20.pkl', 'rb') as f:
    dataa = pickle.load(f)
clfss = dataa['clfs']
X_train = dataa['X_train']; X_test = dataa['X_test']; Y_train = dataa['Y_train']; Y_test = dataa['Y_test']
mdl = clfss['XGB']
# from xgboost import plot_importance
# plot_importance(clf)
f_imp = mdl.feature_importances_
feat_num = -20
# =============================================================================
# plt.bar(range(len(f_imp)), f_imp)
# plt.stem(390 - 1, f_imp[390-1])
# plt.show()
# =============================================================================
idxs = np.argsort(f_imp)
f_names = np.ceil(np.array(range(1, 1281)) / 40)
f_names_orig = np.array(range(1, 1281))
features = X_train
plt.figure(figsize = (10, 10))
plt.title('XGB Feature Importances')
plt.barh(range(len(idxs[feat_num:])), f_imp[idxs[feat_num:]], color='b', align='center')
plt.yticks(range(len(idxs[feat_num:])), [f_names_orig[i] for i in idxs[feat_num:]])
plt.xlabel('Relative Importance')
plt.show()
#%% this 21
mdl = clfss['RF']
# from xgboost import plot_importance
# plot_importance(clf)
f_imp = mdl.feature_importances_
feat_num = -20
# =============================================================================
# plt.bar(range(len(f_imp)), f_imp)
# plt.stem(390 - 1, f_imp[390-1])
# plt.show()
# =============================================================================
idxs = np.argsort(f_imp)
f_names = np.ceil(np.array(range(1, 1281)) / 40)
f_names_orig = np.array(range(1, 1281))
features = X_train
plt.figure(figsize = (10, 10))
plt.title('RF Feature Importances')
plt.barh(range(len(idxs[feat_num:])), f_imp[idxs[feat_num:]], color='b', align='center')
plt.yticks(range(len(idxs[feat_num:])), [f_names_orig[i] for i in idxs[feat_num:]])
plt.xlabel('Relative Importance')
plt.show()

from sklearn.ensemble import RandomForestClassifier
clfr = RandomForestClassifier(max_depth=16, random_state=0)

feats_idx = idxs[feat_num:][::-1]
# f_names = list(range(1, 1281))[feats_idx]
clfr.fit(X_train[:, feats_idx], Y_train)
y_pred_train= clfr.predict(X_train[:, feats_idx])
y_pred_test = clfr.predict(X_test[:, feats_idx])
tracc = accuracy_score(Y_train, y_pred_train)
teacc = accuracy_score(Y_test, y_pred_test)

import shap
# plt.figure(figsize = (10, 10))
samples = X_train[:, feats_idx]

explainer = shap.TreeExplainer(clfr)
shap_values = explainer.shap_values(samples, approximate=False, check_additivity=False)

shap.summary_plot(shap_values, samples, feature_names = np.ceil(feats_idx / 40), max_display = 100)
plt.show()

shap.summary_plot(shap_values, samples, feature_names = feats_idx, max_display = 100)
plt.show()
uik
feats_chosen_shap = np.array([1063, 950, 1029, 936, 1124,659, 1093,177,1192,347,1083,961,875,373,743,330,107,346,55,723])
x_samples = X_train[:, feats_chosen_shap]
clfx = XGBClassifier()
feats_idx = idxs[feat_num:][::-1]
# f_names = list(range(1, 1281))[feats_idx]
clfx.fit(X_train[:, feats_chosen_shap], Y_train)
y_pred_train= clfx.predict(X_train[:, feats_chosen_shap])
y_pred_test = clfx.predict(X_test[:, feats_chosen_shap])
tracc = accuracy_score(Y_train, y_pred_train)
teacc = accuracy_score(Y_test, y_pred_test)
#%%
# =============================================================================
# explainer = shap.KernelExplainer(clfx.predict_proba,X_test[:100, :20])
# testxgb_shap_values = explainer.shap_values(X_test[1:100,0:20], nsamples = 10)
# shap.summary_plot(testxgb_shap_values, samples, feature_names = feats_idx)
# =============================================================================


feats_chosen_shap = np.array([1063, 950, 1029, 936, 1124,659, 1093,177,1192,347,1083,961,875,373,743,330,107,346,55,723])
x_samples = X_train[:, feats_chosen_shap]
clfx = XGBClassifier()
feats_idx = idxs[feat_num:][::-1]
# f_names = list(range(1, 1281))[feats_idx]
clfx.fit(X_train[:, feats_chosen_shap], Y_train)
y_pred_train= clfx.predict(X_train[:, feats_chosen_shap])
y_pred_test = clfx.predict(X_test[:, feats_chosen_shap])
tracc = accuracy_score(Y_train, y_pred_train)
teacc = accuracy_score(Y_test, y_pred_test)

#%% this 2
from xgboost import XGBClassifier
clfx = XGBClassifier()
feats_idx = idxs[feat_num:][::-1]
# f_names = list(range(1, 1281))[feats_idx]
clfx.fit(X_train[:, feats_idx], Y_train)
y_pred_train= clfx.predict(X_train[:, feats_idx])
y_pred_test = clfx.predict(X_test[:, feats_idx])
tracc = accuracy_score(Y_train, y_pred_train)
teacc = accuracy_score(Y_test, y_pred_test)

import shap
# plt.figure(figsize = (10, 10))
samples = X_train[:, feats_idx]

explainer = shap.TreeExplainer(clfx)
shap_values = explainer.shap_values(samples, approximate=False, check_additivity=False)

shap.summary_plot(shap_values, samples, feature_names = np.ceil(feats_idx / 40), max_display = 100)
plt.show()

shap.summary_plot(shap_values, samples, feature_names = feats_idx, max_display = 100)
plt.show()
#%%
# =============================================================================
# explainer = shap.KernelExplainer(clfx.predict_proba,X_test[:100, :20])
# testxgb_shap_values = explainer.shap_values(X_test[1:100,0:20], nsamples = 10)
# shap.summary_plot(testxgb_shap_values, samples, feature_names = feats_idx)
# =============================================================================


feats_chosen_shap = np.array([1063, 950, 1029, 936, 1124,659, 1093,177,1192,347,1083,961,875,373,743,330,107,346,55,723])
x_samples = X_train[:, feats_chosen_shap]
clfx = XGBClassifier()
feats_idx = idxs[feat_num:][::-1]
# f_names = list(range(1, 1281))[feats_idx]
clfx.fit(X_train[:, feats_chosen_shap], Y_train)
y_pred_train= clfx.predict(X_train[:, feats_chosen_shap])
y_pred_test = clfx.predict(X_test[:, feats_chosen_shap])
tracc = accuracy_score(Y_train, y_pred_train)
teacc = accuracy_score(Y_test, y_pred_test)

#%% this 3
from sklearn.decomposition import PCA
pca = PCA(n_components=-feat_num)
xtr = pca.fit_transform(X_train)
xte = pca.transform(X_test)
print(pca.explained_variance_ratio_)
clfx = XGBClassifier()
feats_idx = idxs[feat_num:][::-1]
# f_names = list(range(1, 1281))[feats_idx]
clfx.fit(xtr, Y_train)
y_pred_train= clfx.predict(xtr)
y_pred_test = clfx.predict(xte)
tracc = accuracy_score(Y_train, y_pred_train)
teacc = accuracy_score(Y_test, y_pred_test)
