import pandas as pd
import random
import numpy as np
import sklearn as sk
import sklearn.cluster
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import torch
from torch import nn

import math
import matplotlib.pyplot as plt

data = pd.read_csv("preprocessed.csv")
print(data.shape)

train_ratio = 0.6
validation_ratio = 0.2
test_ratio = 0.2

# train is now 60% of the entire data set
train, proto_test= train_test_split(data, test_size=1 - train_ratio, random_state=17)

# test, validation are now 20% of the initial data set
val, test = train_test_split(proto_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=17)

def collapse(cm):
    '''
    Make all consufiosn matrices 2x2 to show total false positives etc.
    :param confusion matrix, np array
    :return: collapsed np array
    '''
    TP = np.diag(cm).sum()
    TN = np.sum(cm) - np.sum(cm[1, :]) - np.sum(cm[:, 1]) - np.diag(cm)[1]  # TN = total - FP - FN +TP
    FP = np.sum(cm[1, :]) - np.diag(cm)[1]  # Sum of elements in row 1 (actual negative class), excluding TP
    FN = np.sum(cm[:, 1]) - np.diag(cm)[1]  # Sum of elements in column 1 (predicted negative class), excluding TP


    TP, TN, FP, FN = float(TP), float(TN), float(FP), float(FN)

    collapsed_cm = np.array([[TP, FN], [FP, TN]])
    return collapsed_cm

entropies_freq = []
entropies_kmeans = []

accuracies_freq_R = []
accuracies_kmeans_R = []

accuracies_freq_SVM = []
accuracies_kmeans_SVM = []

accuracies_freq_GB =[]
accuracies_kmeans_GB = []


def class_finder(train, val):
    '''

    :param train: pandas dataframe of traning data
    :param val: pandas dataframe of validatiion data
    :return:
    '''

    popularity_train = train.loc[:, train.columns == "popularity"].values
    popularity_val = val.loc[:, val.columns == "popularity"].values
    X_train = train.loc[:, train.columns != "popularity"].values

    X_val = val.loc[:, val.columns != "popularity"].values


    '''
    Creating 3 binning types (constant bin width, constant bin population, kmeans binning) for each number
    of classes in below list.
    '''
    classes =[3,4,5,6,7,8,9,10,11,12]
    for i in classes:

        splitter1 = sklearn.preprocessing.KBinsDiscretizer(subsample = 200000, n_bins=i, random_state=17, encode= 'ordinal', strategy = 'quantile')
        splitter2 = sklearn.preprocessing.KBinsDiscretizer(subsample = 200000, n_bins=i, random_state=17, encode= 'ordinal', strategy = 'uniform')
        splitter3 = sklearn.preprocessing.KBinsDiscretizer(subsample = 200000, n_bins=i, random_state=17, encode= 'ordinal', strategy = 'kmeans')
        frequency = splitter1.fit(popularity_train)
        width = splitter2.fit(popularity_train)
        kmeans = splitter3.fit(popularity_train)
        train_bins1 = frequency.transform(train.loc[:, train.columns == "popularity"].values)
        train_bins2 = width.transform(train.loc[:, train.columns == "popularity"].values)
        train_bins3 = kmeans.transform(train.loc[:, train.columns == "popularity"].values)

        '''Random Forest Classifier'''
        model_test1 = RandomForestClassifier(random_state=17)
        model_test1.fit(X_train, train_bins1.ravel())
        validation_result1 = np.array(model_test1.predict(X_val))

        val_bins1 = frequency.transform(val.loc[:, val.columns == "popularity"].values)

        #confusion matrix 1
        mat1 = sk.metrics.confusion_matrix(val_bins1, validation_result1)

        model_test2 = RandomForestClassifier(random_state=17)
        model_test2.fit(X_train, train_bins2.ravel())
        validation_result2 = np.array(model_test2.predict(X_val))

        val_bins2 = width.transform(val.loc[:, val.columns == "popularity"].values)

        #confusion matrix 2
        mat2 = sk.metrics.confusion_matrix(val_bins2, validation_result2)

        model_test3 = RandomForestClassifier(random_state=17)
        model_test3.fit(X_train, train_bins3.ravel())
        validation_result3 = np.array(model_test3.predict(X_val))

        val_bins3 = kmeans.transform(val.loc[:, val.columns == "popularity"].values)

        #confusion matrix 3
        mat3 = sk.metrics.confusion_matrix(val_bins3, validation_result3)


        '''SVM'''

        # Train a linear SVM model (LINEAR KERNEL)
        svm_model1 = make_pipeline(StandardScaler(), SVC(kernel='linear', decision_function_shape='ovr'))
        svm_model3 = make_pipeline(StandardScaler(), SVC(kernel='linear', decision_function_shape='ovr'))
        svm_model1.fit(X_train, train_bins1.ravel())
        svm_model3.fit(X_train, train_bins3.ravel())
        SVM_pred_val1 = svm_model1.predict(X_val)
        SVM_pred_val3 = svm_model3.predict(X_val)
        SVM_mat1 = sk.metrics.confusion_matrix(val_bins1, SVM_pred_val1)
        SVM_mat3 = sk.metrics.confusion_matrix(val_bins2, SVM_pred_val3)

        '''GBM'''
        # Define GBM models
        gbm_model1 = GradientBoostingClassifier()
        gbm_model3 = GradientBoostingClassifier()

        # Fit GBM models to training data
        gbm_model1.fit(X_train, train_bins1.ravel())
        gbm_model3.fit(X_train, train_bins3.ravel())

        # Predict on validation data
        gbm_pred_val1 = gbm_model1.predict(X_val)
        gbm_pred_val3 = gbm_model3.predict(X_val)

        # Calculate confusion matrices
        gbm_mat1 = sk.metrics.confusion_matrix(val_bins1, gbm_pred_val1)
        gbm_mat3 = sk.metrics.confusion_matrix(val_bins2, gbm_pred_val3)



        # for p in range(i):
        #     print(min(popularity_val[val_bins3 == p]), max(popularity_val[val_bins3 == p]))

        '''
        Trying to use cohen's kappa measurement (didn't turn out useful)
        '''
        # Pe1, Pe2, Pe3 = 0, 0, 0
        # for p in range(i):
        #     Pe1 += (sum(val_bins1 == p)*sum(validation_result1 == p))
        #     Pe2+= (sum(val_bins2 == p) * sum(validation_result2 == p))
        #     Pe3 += (sum(val_bins3 == p) * sum(validation_result3 == p))
        # Pe1 = Pe1 * ((1 / val.shape[0]) ** 2)
        # Pe2 = Pe2 * ((1 / val.shape[0]) ** 2)
        # Pe3 = Pe3 * ((1 / val.shape[0]) ** 2)
        #
        # Po1 = np.trace(mat1)/val.shape[0]
        # Po2 = np.trace(mat2)/val.shape[0]
        # Po3 = np.trace(mat3)/val.shape[0]

        '''calculating entropy'''
        proportion1 = np.bincount(val_bins1.flatten().astype(int))/val.shape[0]
        proportion3 = np.bincount(val_bins3.flatten().astype(int)) / val.shape[0]

        entropy_freq = -1*np.sum(proportion1 * np.log2(proportion1))
        entropy_kmeans = -1*np.sum(proportion3 * np.log2(proportion3))
        entropies_freq.append(entropy_freq)
        entropies_kmeans.append(entropy_kmeans )

        '''Calculating datum accuracy'''
        Binomial_TP = val.shape[0] / i
        #random forest
        accuracies_freq_R.append(np.trace(mat1) - Binomial_TP)
        accuracies_kmeans_R.append(np.trace(mat3) - Binomial_TP)

        #SVM
        accuracies_freq_SVM.append(np.trace(SVM_mat1) - Binomial_TP)
        accuracies_kmeans_SVM.append(np.trace(SVM_mat3) - Binomial_TP)

        #GBM
        accuracies_freq_GB.append(np.trace(gbm_mat1) - Binomial_TP)
        accuracies_kmeans_GB.append(np.trace(gbm_mat3) - Binomial_TP)



        print(f"{i} equal frequency bins:")
        print('class entropy', entropy_freq)
        print("True predictions - binomial true predictions and entropy")
        print('Rforest', accuracies_freq_R[i-3], 'SVM', accuracies_freq_SVM[i-3], 'GBM', accuracies_freq_GB[i-3])

        print(f"{i} kmeans bins:")
        print('class entropy', entropy_kmeans)
        print('Rforest', accuracies_kmeans_R[i-3], 'SVM', accuracies_kmeans_SVM[i-3], 'GBM', accuracies_kmeans_GB[i-3])
        print("\n \n \n")

    '''Entropies Plot'''
    plt.plot(classes, entropies_freq, label = 'Classes with equal freq')
    plt.plot(classes, entropies_kmeans, label = 'Classes with kmeans binning')
    plt.legend()
    plt.xlabel('Number of Classes')
    plt.ylabel('Entropy')
    plt.title('Entropies for Different Binning Methods')

    '''Differenced True Positive Graph for Equal Frequencies'''
    plt.figure()
    plt.xlabel('Number of Classes')
    plt.ylabel('Difference between number of Correct Classifications and Random' '\n' 'Correct Guesses (n*1/(class num.), Likely Correct Binomial Guesses)')
    plt.title('Differenced True Positive Graph for Equal Frequencies')
    plt.plot(classes, accuracies_freq_R, label = 'Random Forest')
    plt.plot(classes, accuracies_freq_SVM, label = 'SVM')
    plt.plot(classes, accuracies_freq_GB, label='GBM')
    plt.legend()

    '''Differenced True Positive Graph for Kmeans Binning'''
    plt.figure()
    plt.xlabel('Number of Classes')
    plt.ylabel(
        'Difference between number of Correct Classifications and Random' '\n' 'Correct Guesses (n*1/(class num.), Likely Correct Binomial Guesses)')
    plt.title('Differenced True Positive Graph for Kmeans Binning')
    plt.plot(classes, accuracies_kmeans_R, label='Random Forest')
    plt.plot(classes, accuracies_kmeans_SVM, label='SVM')
    plt.plot(classes, accuracies_kmeans_GB, label='GBM')
    plt.legend()

    '''Product plot for frequency bins'''
    plt.figure()
    plt.xlabel('Number of Classes')
    plt.ylabel('Product of Entropy and Differenced True Classifications')
    plt.title('Entropy x Accuracy for Equal Frequency Bins')
    efreq = np.array(entropies_freq)
    a_SVM_freq = np.array(accuracies_freq_SVM)
    a_R_freq = np.array(accuracies_freq_R)
    a_GB_freq = np.array(accuracies_freq_GB)

    plt.plot(classes, efreq*a_R_freq, label = 'Random Forest')
    plt.plot(classes, efreq*a_SVM_freq, label = 'SVM')
    plt.plot(classes, efreq * a_GB_freq, label='GBM')
    plt.legend()

    '''Product plot for kmeans bins'''
    plt.figure()
    plt.xlabel('Number of Classes')
    plt.ylabel('Product of Entropy and Differenced True Classifications')
    plt.title('Entropy x Accuracy for Kmeans Bins')
    ek = np.array(entropies_kmeans)
    a_SVM_kmeans = np.array(accuracies_kmeans_SVM)
    a_R_kmeans = np.array(accuracies_kmeans_R)
    a_GB_kmeans = np.array(accuracies_kmeans_GB)

    plt.plot(classes, ek * a_R_kmeans, label='Random Forest')
    plt.plot(classes, ek * a_SVM_kmeans, label='SVM')
    plt.plot(classes, ek * a_GB_kmeans, label='GBM')
    plt.legend()

    plt.show()


class_finder(train, val)

'''
code for implementing manually constructed classes
'''
# X_train = train_ready.loc[:, train_ready.columns != "popularity"].values
# Y_train = train_ready.loc[:, train_ready.columns == "popularity"].values
#
# X_val = val_ready.loc[:, val_ready.columns != "popularity"].values
# Y_val = val_ready.loc[:, val_ready.columns == "popularity"].values
# print(Y_val)
#
# # Build and run model
# model_test1 = RandomForestClassifier(random_state=17)
# model_test1.fit(X_train, Y_train.ravel())
# result = np.array(model_test1.predict(X_val))
#
# # Display number of incorrectly predicted classes from validation set
# print('Number of incorrectly classified songs:', sum(Y_val.ravel() != result))
# differences = (Y_val.ravel()[np.where(Y_val.ravel() != result)]-result[np.where(Y_val.ravel() != result)])**2
# print('Average difference in class for incorrectly classified songs:', math.sqrt(sum(differences)/sum(Y_val.ravel() != result)))
#


