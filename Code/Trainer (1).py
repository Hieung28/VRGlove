from builtins import print

import csv
import numpy as np
from timeit import default_timer as timer
# from sklearn.datasets import make_multilabel_classification, make_classification
# from sklearn.decomposition import PCA
# from sklearn.cross_decomposition import CCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib

# #generating data to test models
# print("Generating datasets")
# start = timer()
# # X, y = make_multilabel_classification(n_samples=20000, n_classes=10,
# #                                       n_features = 30,
# #                                       n_labels=10,
# #                                       allow_unlabeled=False,
# #                                       random_state=1)
# X, y = make_classification(n_samples=20000, n_features=30,
#                            n_informative=28, n_redundant=2,
#                            n_classes=10, hypercube=False,
#                            random_state=np.random)
# end = timer()
# print("Generating done. Time elapsed:", end-start, " seconds")

#acquiring data from external csv file gathered from data glove
myData = np.genfromtxt('trainingData.csv', delimiter=',')
# print(myData.shape)

#splitting data for training
X = myData[:, 0:myData.shape[1]-1]
#splitting labels for training
y = myData[:, -1]
# y=y.astype('int')
# X = preprocessing.normalize(preprocessing.scale(X))

#Splitting data in to a training set and a test set to see the accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=0)
#initiate classifier object, using One Vs Rest method
classif = OneVsRestClassifier(SVC(C=1,
                                  kernel='poly', degree=5,
                                  coef0=0.5,
                                  tol=1e-4,
                                  shrinking=True,
                                  gamma='scale'))
#begin training and testing
print("Training...")
start = timer()
classif.fit(X_train, y_train)
end = timer()
print(X_train)
print("Training done. Time elapsed:", end-start, " seconds")
print("Testing...")
train_score = classif.score(X_test, y_test)
train_score = format(train_score*100, '.2f')
print('Testing done. Training accuracy:', train_score, "%")
# Save model to disk
filename = 'finalizedModel.sav'
print("Saving model to disk under name:", filename, "...")
joblib.dump(classif, filename)
print("Saving done. \n")

load_model=joblib.load(filename)

