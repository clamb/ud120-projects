#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
# clf = svm.SVC(kernel='linear')
clf = svm.SVC(kernel='rbf', gamma='auto', C=10000.)

# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

t0 = time()
clf.fit(features_train, labels_train)
print "Time to fit: ", round(time() - t0, 3), "s"

t0 = time()
clf.predict(features_test)
print "Time to predict: ", round(time() - t0, 3), "s"

print "10th: ", clf.predict([features_test[10]])
print "26th: ", clf.predict([features_test[26]])
print "50th: ", clf.predict([features_test[50]])

print "Score: ", clf.score(features_test, labels_test)

count = 0
for feature_test in features_test:
    if clf.predict([feature_test]) == 1:
        count += 1

print "Chris(1) appears: ", count, " times"


#########################################################


