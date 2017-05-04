#
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

data = pd.read_csv('baidu_user34.csv', delimiter=',')
kfold = 3
skf = StratifiedKFold(n_splits=kfold)
nCols = data.shape[1]
#extract attributes and class target
X = data.iloc[:,1:nCols-1].as_matrix()
Y = data.iloc[:,2].as_matrix()
#cross validation
for train, test in skf.split(X, Y):
    trainX = X[train,:] 
    testX = X[test,:]
    trainY = Y[train].reshape((len(train),1))
    testY = Y[test].reshape((len(test),1))
    train = np.append(trainX, trainY, axis=1)
    test = np.append(testX, testY, axis=1)
