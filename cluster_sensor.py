import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from cluster_func import kmeans
from cluster_func import em


data_full = pd.read_csv('./Sensorless_drive_diagnosis.csv', header=None)

#Randomly sample the data to reduce the size of dataset due to computation difficulty
RandInd = np.random.choice(len(data_full),30000)
data = data_full.iloc[RandInd,:].reset_index().drop(['index'], axis = 1)

X = data.iloc[:,:-1]
y = data.iloc[:,-1] - 1

#Splitting data into training and testing and keeping testing data aside
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

#Converting into numpy arrays
X_train = X_train.as_matrix()
X_test = X_test.as_matrix()
y_train = y_train.as_matrix() - 1
y_test = y_test.as_matrix() - 1


#Preprocessing the data between 0 and 1
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

means_init = np.array([X[y == i].mean(axis=0) for i in range(11)])

##############################################################################################################################
#For Expected Maximization
_ = em(X_train, X_test, y_train, y_test, init_means = means_init, component_list = [3,4,5,6,7,8,9,10,11,12, 13, 14, 15, 16], num_class =11, toshow = 1)

#############################################################################################################################
#For KMeans
_=  kmeans(X_train, X_test, y_train, y_test, init_means = means_init, component_list = [3,4,5,6,7,8,9,10,11, 12,13,14,15,16], num_class = 11, toshow = 1)