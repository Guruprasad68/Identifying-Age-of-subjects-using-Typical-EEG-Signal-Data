#%% import packages
import numpy as np, pandas as pd, matplotlib.pyplot as plt, os, time, pickle, librosa as lr
from scipy.stats import ttest_ind
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import random
#%% code

# =============================================================================
# X_train, val, test will have shapes num_subjects * num_segments, rmse dimension (40) * numchannels (32)
# =============================================================================

# modify paths and script as necessary
path = 'C:/UWMad/Subjects/F21/ECE539/course_project/'
only32 = True
if only32 == True:
    with open(path + 'data/train32.pkl', 'rb') as f:
        L = pickle.load(f)
else:
    L = os.listdir(path+'data/data_eeg_age_v1/data2kaggle/train/')
data_path = path+'data/data_eeg_age_v1/data2kaggle/'
dataset = 'train/'
bs = 23; bstr = 17; bste = 6 

r = random.sample(range(bs), bs) # randomly picking tain and test segments from each signal so as to avoid any temporal dependencies
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
    for i in range(0, df.to_numpy().shape[1] - 4):
        xtr = []
        xte = []
        X = df.to_numpy()[:, i][0:230000]
        ns = int(X.shape[0] / bs)
        for ih in r:
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
        ktr = 1
        kte = 1
    if L.index(items) == 0:
        X_train = Xtr#[:, :, np.newaxis]
        X_test = Xte#[:, :, np.newaxis]
        Y_train = np.array(ytr)[:, np.newaxis]
        Y_test = np.array(yte)[:, np.newaxis]
        # uik
    else:
        X_train = np.concatenate((X_train, Xtr), axis = 0)
        X_test = np.concatenate((X_test, Xte), axis = 0)
        Y_train = np.concatenate((Y_train, np.array(ytr)[:, np.newaxis]), axis = 0)
        Y_test = np.concatenate((Y_test, np.array(yte)[:, np.newaxis]), axis = 0)
    print('[info 3]', L.index(items), len(L), X_train.shape, Y_train.shape, X_test.shape, Y_test.shape, len(ytr), len(yte))
    # UIK
# uik
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)#[:, :, np.newaxis]
X_test = scaler.transform(X_test)#[:, :, np.newaxis]
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.20, stratify = Y_train, shuffle = True)
data32 = {'x_train': X_train, 'x_val': X_val, 'x_test': X_test, 'y_train': Y_train, 'y_val': Y_val, 'y_test': Y_test}
with open('C:/UWMad/Subjects/F21/ECE539/course_project/data/Trdata32_CNN1D_200.pkl', 'wb') as f:
    pickle.dump(data32, f)
    
    
    
import sys
sys.exit()

#%% import packages
import numpy as np, pandas as pd, matplotlib.pyplot as plt, os, time, pickle, librosa as lr
from scipy.stats import ttest_ind
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import random
#%% code

# =============================================================================
# X_train, val, test will have shapes num_subjects * num_segments, rmse dimension (40) * numchannels (32)
# =============================================================================

# modify paths and script as necessary
path = 'C:/UWMad/Subjects/F21/ECE539/course_project/'
only32 = True
if only32 == True:
    with open(path + 'data/eval32.pkl', 'rb') as f:
        L = pickle.load(f)
else:
    L = os.listdir(path+'data/data_eeg_age_v1/data2kaggle/train/')
data_path = path+'data/data_eeg_age_v1/data2kaggle/'
dataset = 'eval/'
bs = 23; bstr = 17; bste = 6 

r = random.sample(range(bs), bs) # randomly picking tain and test segments from each signal so as to avoid any temporal dependencies
for items in L:
    df = pd.read_csv(data_path + dataset+ items, skiprows = [0, 1])
    h = pd.read_csv(data_path + dataset+ items, index_col=0, nrows=0).columns.tolist()[0]
    h = int(h.split('= ')[1])
    h_ = np.argmax([10<=h<=30, 31<=h<=40, 41<=h<=50, 51<=h<=60, 61<=h<=90])
    vals_tr = []
    vals_te = []
    yte1 = []
    yte2 = []
    yte3 = []
    yte4 = []

    ktest = 0
    ktrain = 0
    kte1 = 0
    kte2 = 0
    kte3 = 0
    kte4 = 0
    for i in range(0, df.to_numpy().shape[1] - 4):
        xte1 = []
        xte2 = []
        xte3 = []
        xte4 = []
        X = df.to_numpy()[:, i][0:230000]
        ns = int(X.shape[0] / bs)
        for ih in r:
            temp = lr.feature.rms(X[ih*ns : (ih+1)*ns], frame_length = 512, hop_length = 256)
            if len(xte1) <= 5:
                xte1.append(temp.squeeze())
                if kte1 == 0:
                    yte1.append(h_)
            elif len(xte2) <= 5:
                xte2.append(temp.squeeze())
                if kte2 == 0:
                    yte2.append(h_)
            elif len(xte3) <= 5:
                xte3.append(temp.squeeze())
                if kte3 == 0:
                    yte3.append(h_)
            elif len(xte4) <= 4:
                xte4.append(temp.squeeze())
                if kte4 == 0:
                    yte4.append(h_)
        if i == 0:
            Xte1 = np.array(xte1)
            Xte2 = np.array(xte2)
            Xte3 = np.array(xte3)
            Xte4 = np.array(xte4)
        else:
            Xte1 = np.concatenate((Xte1, np.array(xte1)), axis = 1)
            Xte2 = np.concatenate((Xte2, np.array(xte2)), axis = 1)
            Xte3 = np.concatenate((Xte3, np.array(xte3)), axis = 1)
            Xte4 = np.concatenate((Xte4, np.array(xte4)), axis = 1)
        kte1 = 1
        kte2 = 1
        kte3 = 1
        kte4 = 1
    if L.index(items) == 0:
        X_test1 = Xte1#[:, :, np.newaxis]
        X_test2 = Xte2#[:, :, np.newaxis]
        X_test3 = Xte3#[:, :, np.newaxis]
        X_test4 = Xte4#[:, :, np.newaxis]
        
        Y_test1 = np.array(yte1)[:, np.newaxis]
        Y_test2 = np.array(yte2)[:, np.newaxis]
        Y_test3 = np.array(yte3)[:, np.newaxis]
        Y_test4 = np.array(yte4)[:, np.newaxis]
        # uik
    else:
        X_test1 = np.concatenate((X_test1, Xte1), axis = 0)
        X_test2 = np.concatenate((X_test2, Xte2), axis = 0)
        X_test3 = np.concatenate((X_test3, Xte3), axis = 0)
        X_test4 = np.concatenate((X_test4, Xte4), axis = 0)
        Y_test1 = np.concatenate((Y_test1, np.array(yte1)[:, np.newaxis]), axis = 0)
        Y_test2 = np.concatenate((Y_test2, np.array(yte2)[:, np.newaxis]), axis = 0)
        Y_test3 = np.concatenate((Y_test3, np.array(yte3)[:, np.newaxis]), axis = 0)
        Y_test4 = np.concatenate((Y_test4, np.array(yte4)[:, np.newaxis]), axis = 0)
        # uik
    print('[info 3]', L.index(items), len(L), X_test1.shape, X_test2.shape, X_test3.shape, X_test4.shape, Y_test1.shape, Y_test2.shape, Y_test3.shape, Y_test4.shape)
    # UIK
uik
with open('C:/UWMad/Subjects/F21/ECE539/course_project/data/Trdata32_CNN1D.pkl', 'rb') as f:
    dataa=pickle.load(f)
    
x_train = dataa['x_train'].squeeze()
scaler = StandardScaler()
scaler.fit(x_train)#[:, :, np.newaxis]
X_test1 = scaler.transform(X_test1)#[:, :, np.newaxis]
X_test2 = scaler.transform(X_test2)#[:, :, np.newaxis]
X_test3 = scaler.transform(X_test3)#[:, :, np.newaxis]
X_test4 = scaler.transform(X_test4)#[:, :, np.newaxis]
print(X_test1.shape, X_test2.shape, X_test3.shape, X_test4.shape, Y_test1.shape, Y_test2.shape, Y_test3.shape, Y_test4.shape)
# X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.20, stratify = Y_train, shuffle = True)

data32 = {'X_test1': X_test1, 'X_test2': X_test2, 
          'X_test3': X_test1, 'X_test4': X_test4,
          'Y_test1': Y_test1, 'Y_test2': Y_test2, 
          'Y_test3': Y_test3, 'Y_test4': Y_test4}
with open('C:/UWMad/Subjects/F21/ECE539/course_project/data/Evdata32_CNN1D_test.pkl', 'wb') as f:
    pickle.dump(data32, f)