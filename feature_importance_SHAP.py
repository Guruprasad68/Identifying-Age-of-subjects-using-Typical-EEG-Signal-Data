#%% import packages
import numpy as np, pandas as pd, matplotlib.pyplot as plt, os, time, pickle, librosa as lr
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import random
np.random.seed(1)

#%%
with open('___/datafile.pkl', 'rb') as f:
    dataa = pickle.load(f)
clfss = dataa['clfs']
X_train = dataa['X_train']; X_test = dataa['X_test']; Y_train = dataa['Y_train']; Y_test = dataa['Y_test']
#%%
mdl = clfss['XGB']

f_imp = mdl.feature_importances_# obtaining features' Gini importance scores from XGBoost model
feat_num = -100

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

from xgboost import XGBClassifier
clfx = XGBClassifier()
feats_idx = idxs[feat_num:][::-1]
clfx.fit(X_train[:, feats_idx], Y_train)
y_pred_train= clfx.predict(X_train[:, feats_idx])
y_pred_test = clfx.predict(X_test[:, feats_idx])
tracc = accuracy_score(Y_train, y_pred_train)
teacc = accuracy_score(Y_test, y_pred_test)
print('xgb f_imp order:', tracc, teacc)
import shap
samples = X_train[:, feats_idx]

explainer = shap.TreeExplainer(clfx) # ordering features using SHAP
shap_values = explainer.shap_values(samples, approximate=False, check_additivity=False)


shap.summary_plot(shap_values, samples, feature_names = np.ceil(feats_idx / 40), max_display = 20)
plt.show()

shap_values1 = shap_values.copy()
# manually picking SHAP ordered features from the previous plot
feats_chosen_shap = np.array([950, 937, 931, 1068,1075,
                              1152, 1148, 1063, 1029, 1066, 819,
                              1064, 1124, 951, 807, 961, 1034,
                              824, 936, 1010, 338, 920, 1115,
                              790, 1228, 1102, 1050, 228, 733, 1013, 346, 1001, 333, 1189,
                              355, 330, 347, 798, 490, 363, 659, 1192, 107,
                              629, 187, 1223, 726, 875, 55, 1083, 1093,129,
                              1169, 177, 270, 368, 1218, 412, 675, 744, 252, 663,
                              866, 422, 800, 387, 534, 36, 124, 677, 51, 163, 1193,
                              578, 189, 353, 544, 1252, 403, 535, 495, 511, 723, 389,
                              902, 415, 678, 743, 528, 166, 576, 428, 676, 496, 854,
                              555, 414, 373, 863, 64])
fl_15 = []
for i in range(0, 5):
    fl_15.append(np.concatenate((shap_values1[i][:, 0:15], shap_values1[i][:, -15:None]), axis = 1))
x_samples = np.concatenate((X_train[:, feats_chosen_shap[0:15]], X_train[:, feats_chosen_shap[-15:None]]), axis = 1)
shap.summary_plot(fl_15, x_samples, feature_names = np.ceil(feats_idx / 40), max_display = 100)

x_samples = X_train[:, feats_chosen_shap]
clfx = XGBClassifier()
feats_idx = idxs[feat_num:][::-1]
clfx.fit(X_train[:, feats_chosen_shap], Y_train)
y_pred_train= clfx.predict(X_train[:, feats_chosen_shap])
y_pred_test = clfx.predict(X_test[:, feats_chosen_shap])
tracc = accuracy_score(Y_train, y_pred_train)
teacc = accuracy_score(Y_test, y_pred_test)
print('xgb shap f_imp order:', tracc, teacc)
#%%
mdl = clfss['RF']

f_imp = mdl.feature_importances_ # obtaining features' Gini importance score from Random Forest model
feat_num = -100

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
print('RF f_imp order:', tracc, teacc)
import shap
# plt.figure(figsize = (10, 10))
samples = X_train[:, feats_idx]

explainer = shap.TreeExplainer(clfx) # ordering features using SHAP
shap_values = explainer.shap_values(samples, approximate=False, check_additivity=False)

# =============================================================================
# shap.summary_plot(shap_values, samples, feature_names = np.ceil(feats_idx / 40), max_display = 100)
# plt.show()
# =============================================================================

shap.summary_plot(shap_values, samples, feature_names = np.ceil(feats_idx / 40), max_display = 20)
plt.show()

# manually picking SHAP ordered features from the previous plot
feats_chosen_shap = np.array([927, 958, 950, 956, 937, 920, 930, 1079, 957,
                              944, 939, 925, 1029, 934, 951, 1078, 1051, 1143, 947,
                              945, 926, 952, 1043, 1073, 932, 1144, 948, 346, 954, 936,
                              922, 1146, 1040, 1130, 935, 1070, 1076, 940, 1069, 1140, 
                              953, 1042, 1064, 938, 1158, 949, 1062, 959, 1053, 942,
                              1155, 1033, 928, 1072, 1151, 946, 1055, 931, 1148, 1034, 1038,
                              1047, 1049, 1030, 929, 1152, 1026, 921, 1074, 1063, 1067, 1031,
                              1036, 933, 333, 1132, 1141, 1071, 943, 1060, 924, 1024, 1122, 1022, 1121,
                              941, 1056, 955, 1156, 1016, 1028, 1065, 1127, 1153, 1046, 1066, 1131, 923,
                              1050, 983])
shap_values1 = shap_values.copy()

fl_15 = []
for i in range(0, 5):
    fl_15.append(np.concatenate((shap_values1[i][:, 0:20], shap_values1[i][:, -20:None]), axis = 1))
x_samples = np.concatenate((X_train[:, feats_chosen_shap[0:20]], X_train[:, feats_chosen_shap[-20:None]]), axis = 1)
shap.summary_plot(fl_15, x_samples, feature_names = np.ceil(feats_idx / 40), max_display = 100)
    

x_samples = X_train[:, feats_chosen_shap]
from sklearn.ensemble import RandomForestClassifier
clfr = RandomForestClassifier(max_depth=16, random_state=0)
feats_idx = idxs[feat_num:][::-1]
# f_names = list(range(1, 1281))[feats_idx]
clfx.fit(X_train[:, feats_chosen_shap], Y_train)
y_pred_train= clfx.predict(X_train[:, feats_chosen_shap])
y_pred_test = clfx.predict(X_test[:, feats_chosen_shap])
tracc = accuracy_score(Y_train, y_pred_train)
teacc = accuracy_score(Y_test, y_pred_test)
print('rf shap f_imp order:', tracc, teacc)
#%%
# =============================================================================
# explainer = shap.KernelExplainer(clfx.predict_proba,X_test[:100, :20])
# testxgb_shap_values = explainer.shap_values(X_test[1:100,0:20], nsamples = 10)
# shap.summary_plot(testxgb_shap_values, samples, feature_names = feats_idx)
# =============================================================================
#%% this 3
from sklearn.decomposition import PCA
pca = PCA(n_components=-feat_num)
xtr = pca.fit_transform(X_train)
xte = pca.transform(X_test)
print(pca.explained_variance_ratio_)
from xgboost import XGBClassifier
clfx = XGBClassifier()
from sklearn.ensemble import RandomForestClassifier
clfr = RandomForestClassifier(max_depth=16, random_state=0)


feats_idx = idxs[feat_num:][::-1]
# f_names = list(range(1, 1281))[feats_idx]
clfx.fit(xtr, Y_train)
y_pred_train= clfx.predict(xtr)
y_pred_test = clfx.predict(xte)
tracc = accuracy_score(Y_train, y_pred_train)
teacc = accuracy_score(Y_test, y_pred_test)
print('xgb pca feat_num:', tracc, teacc)
clfr.fit(xtr, Y_train)
y_pred_train= clfr.predict(xtr)
y_pred_test = clfr.predict(xte)
tracc = accuracy_score(Y_train, y_pred_train)
teacc = accuracy_score(Y_test, y_pred_test)
print('rf pca feat_num:', tracc, teacc)