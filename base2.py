#!/usr/bin/env python
import os
import itertools
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

'''
Task: We remove the emoji from the sequence of
tokens and use it as a label both for training and
testing. The task for our machine learning models
is to predict the single emoji that appears in the
input tweet.
'''

def extractInfo(fname):
	tweets = []
	labels = []
	with open(fname) as fid:
		lines = fid.readlines()
		for line in lines:
			line = line.strip()
			labels.append(line.rsplit(' ',1)[1].strip())
			tweets.append(line.rsplit(' ',1)[0])

	return tweets, labels

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='none', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def encode_labels(labels):
    c = collections.Counter(labels)
    a = {} 
    j = 0  
    for i in c.most_common(len(c)):
        a[i[0]] = j
        j = j+1
    return a
    
def encode(labels, encoding):
    new_labels = list()
    for i in labels:
        new_labels.append(encoding[i])
    
    return new_labels
    
def performance(y_true, y_pred, metric):
	performance = {}
	performance["f1_score"] = metrics.f1_score(y_true, y_label)
	performance["precision"] = metrics.precision_score(y_true, y_label)
    # calculate sensitivity and specificity
	cm = metrics.confusion_matrix(y_true, y_label)
    #total = sum(sum(cm))
    #accuracy = (cm[0,0] + cm[1,1])/total
	sensitivity = cm[1,1]/(cm[1,0] + cm[1,1])
	specificity = cm[0,0]/(cm[0,0] + cm[0,1])
	performance["sensitivity"] = sensitivity
	performance["specificity"] = specificity
	return performance[metric]

def main():
    K = int(input("Enter number of frequent emojis: "))
	# read tweets and labels
    ftrain = "data/"+str(K)+"_train"
    ftest = "data/"+str(K)+"_test"
    tweets_train, labels_train = extractInfo(ftrain)
    tweets_test, labels_test = extractInfo(ftest)
	
    encoding = encode_labels(labels_train)
	
    labels_train_numeric = encode(labels_train, encoding)
    labels_test_numeric = encode(labels_test, encoding)
	
    vectorizer = TfidfVectorizer()
    vectorizer.fit(tweets_train)
	
	# document term matrix
    tweets_train_dtm = vectorizer.transform(tweets_train)

    print(tweets_train_dtm.shape)
	
    tweets_test_dtm = vectorizer.transform(tweets_test)
	
	# creating and training logistic regression model
    logreg = LogisticRegression(penalty="l2", multi_class="ovr")
    logreg.fit(tweets_train_dtm, labels_train_numeric)
    y_predicted = logreg.predict(tweets_test_dtm)  # predicting
			
    print(confusion_matrix(labels_test_numeric, y_predicted))
    class_names = [i for i in range(0,K)]
    cnf_matrix = confusion_matrix(labels_test_numeric, y_predicted)
    np.set_printoptions(precision=4)

	# plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names)
    figname = "Fig_"+str(K)+".png"
    plt.savefig(figname)
    plt.show()
	
    print("Accuracy: ", accuracy_score(labels_test_numeric, y_predicted))
    print("F1-Score: ", f1_score(labels_test_numeric, y_predicted, average="weighted"))
    print("Precision: ", precision_score(labels_test_numeric, y_predicted, average="weighted"))
    print("Recall: ", recall_score(labels_test_numeric, y_predicted, average="weighted"))

if __name__ == "__main__":
	main()

