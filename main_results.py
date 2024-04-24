import pandas as pd
import random
import numpy as np
import sklearn as sk
import sklearn.cluster
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_hastie_10_2
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.inspection import partial_dependence
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import cross_val_score

import math
import matplotlib.pyplot as plt

#data = pd.read_csv("preprocessed.csv") #total data set
data = pd.read_csv("recentpreprocessed.csv") #two most recent decades, uncomment to get feature importance plot without decade.

train_ratio = 0.6
validation_ratio = 0.2
test_ratio = 0.2

# train is now 60% of the entire data set
train, proto_test= train_test_split(data, test_size=1 - train_ratio, random_state=17)

# test, validation are now 20% of the initial data set
val, test = train_test_split(proto_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=17)


'''REDUNDANT - initialising lists to plot stats for choosing classes and initial screening of models'''
# entropies_freq = []
# entropies_kmeans = []
#
# accuracies_freq_R = []
# accuracies_kmeans_R = []
#
# accuracies_freq_SVM = []
# accuracies_kmeans_SVM = []
#
# accuracies_freq_GB =[]
# accuracies_kmeans_GB = []
#
# accuracies_freq_lin =[]
# accuracies_kmeans_lin = []

def calculate_accuracy(confusion_matrix):
    # Calculate total correct predictions
    total_correct = np.sum(np.diag(confusion_matrix))

    # Calculate total predictions
    total_predictions = np.sum(confusion_matrix)

    # Calculate accuracy
    accuracy = total_correct / total_predictions

    return accuracy

def class_finder(train, val, test):
    '''

    :param train: pandas dataframe of traning data
    :param val: pandas dataframe of validatiion data
    :return:
    '''

    popularity_train = train.loc[:, train.columns == "popularity"].values
    popularity_val = val.loc[:, val.columns == "popularity"].values
    popularity_test = test.loc[:, test.columns == "popularity"].values
    X_train = train.loc[:, train.columns != "popularity"].values

    X_val = val.loc[:, val.columns != "popularity"].values
    X_test = test.loc[:, test.columns != "popularity"].values


    '''
    Creating 3 binning types (constant bin width, constant bin population, kmeans binning) for each number
    of classes in below list.
    '''
    classes_choosing = [3,4,5,6,7,8,9,10,11,12] # used initiallyi in loop to select number of classes.
    classes =[4,8]
    for i in classes:


        '''REDUNDANT-  code used for evaluating efficacy of quantile and uniform binning methods'''
        # splitter1 = sklearn.preprocessing.KBinsDiscretizer(subsample = 200000, n_bins=i, random_state=17, encode= 'ordinal', strategy = 'quantile')
        # frequency = splitter1.fit(popularity_train)
        # splitter2 = sklearn.preprocessing.KBinsDiscretizer(subsample = 200000, n_bins=i, random_state=17, encode= 'ordinal', strategy = 'uniform')
        # width = splitter2.fit(popularity_train)
        #
        # train_bins1 = frequency.transform(train.loc[:, train.columns == "popularity"].values)
        # train_bins2 = width.transform(train.loc[:, train.columns == "popularity"].values)

        #
        # '''Random Forest Classifier'''
        # model_test1 = RandomForestClassifier(random_state=17)
        # model_test1.fit(X_train, train_bins1.ravel())
        # validation_result1 = np.array(model_test1.predict(X_val))
        #
        # val_bins1 = frequency.transform(val.loc[:, val.columns == "popularity"].values)
        #
        # # confusion matrix 1
        # mat1 = sk.metrics.confusion_matrix(val_bins1, validation_result1)
        #
        # model_test2 = RandomForestClassifier(random_state=17)
        # model_test2.fit(X_train, train_bins2.ravel())
        # validation_result2 = np.array(model_test2.predict(X_val))
        #
        # val_bins2 = width.transform(val.loc[:, val.columns == "popularity"].values)
        #
        # # confusion matrix 2
        # mat2 = sk.metrics.confusion_matrix(val_bins2, validation_result2)
        #
        # '''SVM'''
        #
        # # Train a linear SVM model (LINEAR KERNEL)
        # svm_model1 = make_pipeline(StandardScaler(), SVC(kernel='linear', decision_function_shape='ovr'))
        #
        # svm_model1.fit(X_train, train_bins1.ravel())
        #
        # SVM_pred_val1 = svm_model1.predict(X_val)
        #
        # SVM_mat1 = sk.metrics.confusion_matrix(val_bins1, SVM_pred_val1)
        #
        # '''GBM'''
        # # Define GBM models
        # gbm_model1 = GradientBoostingClassifier()
        #
        # # Fit GBM models to training data
        # gbm_model1.fit(X_train, train_bins1.ravel())
        # gbm_pred_val1 = gbm_model1.predict(X_val)
        # gbm_mat1 = sk.metrics.confusion_matrix(val_bins1, gbm_pred_val1)
        #
        # '''
        # Linear
        # SGD
        # classifier
        # '''
        # # Define SGD classifier models
        # sgd_model1 = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
        #
        # # Fit SGD classifier models to training data
        # sgd_model1.fit(X_train, train_bins1.ravel())
        #
        # # Predict on validation data
        # sgd_pred_val1 = sgd_model1.predict(X_val)
        #
        # # Calculate confusion matrices
        # sgd_mat1 = sk.metrics.confusion_matrix(val_bins1, sgd_pred_val1)

        '''REDUNDANT - Code used for choosing binning method and number of bins and hyperparam tuning'''
        # splitter3 = sklearn.preprocessing.KBinsDiscretizer(subsample = 200000, n_bins=i, random_state=17, encode= 'ordinal', strategy = 'kmeans')
        # kmeans = splitter3.fit(popularity_train)
        #
        # train_bins3 = kmeans.transform(train.loc[:, train.columns == "popularity"].values)
        #
        #
        # model_test3 = RandomForestClassifier(random_state=17)
        # model_test3.fit(X_train, train_bins3.ravel())
        # validation_result3 = np.array(model_test3.predict(X_val))
        #
        # val_bins3 = kmeans.transform(val.loc[:, val.columns == "popularity"].values)
        #
        # #confusion matrix 3
        # mat3 = sk.metrics.confusion_matrix(val_bins3, validation_result3)
        #
        #
        # svm_model3 = make_pipeline(StandardScaler(), SVC(kernel='linear', decision_function_shape='ovr'))
        # svm_model3.fit(X_train, train_bins3.ravel())
        # SVM_pred_val3 = svm_model3.predict(X_val)
        # SVM_mat3 = sk.metrics.confusion_matrix(val_bins3, SVM_pred_val3)
        #
        #
        # gbm_model3 = GradientBoostingClassifier()
        # gbm_model3.fit(X_train, train_bins3.ravel())
        # gbm_pred_val3 = gbm_model3.predict(X_val)
        #
        # # Calculate confusion matrices
        #
        # gbm_mat3 = sk.metrics.confusion_matrix(val_bins3, gbm_pred_val3)
        #
        #
        # sgd_model3 = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
        # sgd_model3.fit(X_train, train_bins3.ravel())
        # sgd_pred_val3 = sgd_model3.predict(X_val)
        # sgd_mat3 = sk.metrics.confusion_matrix(val_bins3, sgd_pred_val3)


        '''REDUNDANT - Code used for Hyperparam tuning'''
        # X_train = np.vstack((X_train, X_val))
        # train_bins3 = np.vstack((train_bins3, val_bins3))
        #
        # #Define search space for random forest
        # rf_param_dist = {
        #     'n_estimators': Integer(10, 200),
        #     'max_depth': Integer(3, 10),
        #     'min_samples_split': Integer(2, 20),
        #     'min_samples_leaf': Integer(1, 10),
        #     'max_features': ['sqrt', 'log2', None],
        #     'bootstrap': [True, False]
        # }
        #
        # #Defining Search Space for SVM (reduced compared to RF and GBM since programme ran indefinitely otherwise
        # param_grid = {
        #     'C': [0.1, 1, 10],
        #     'kernel': ['linear', 'rbf'],
        #     'gamma': ['scale', 'auto', 0.1, 1.0],
        #     'degree': [2, 3, 4],
        #     'bootstrap': [True, False]
        # } #polynomial kernel was also tested but the loop ran too slowly if all three kernel types were tested at once
        #
        #
        # # Define the search space for GradientBoostingClassifier
        # gbm_param_dist = {
        #     'n_estimators': Integer(10, 200),
        #     'learning_rate': Real(1e-6, 1e+1, prior='log-uniform'),
        #     'max_depth': Integer(3, 10),
        #     'min_samples_split': Integer(2, 20),
        #     'min_samples_leaf': Integer(1, 15),
        #     'max_features': ['sqrt', 'log2', None]
        # }
        #
        # # Perform Bayesian search for RandomForestClassifier
        # rf_bayes_search = BayesSearchCV(RandomForestClassifier(random_state=17),
        #                                 search_spaces=rf_param_dist,
        #                                 n_iter=10, cv=5, random_state=42)
        #
        # rf_bayes_search.fit(X_train, train_bins3.ravel())
        #
        # best_score = 0
        # best_params = None
        #
        # # Loop over each combination of hyperparameters
        # for C in param_grid['C']:
        #     for kernel in param_grid['kernel']:
        #         for gamma in param_grid['gamma']:
        #             for degree in param_grid['degree']:
        #                 # Train SVM model with current hyperparameters
        #                 svm_modeloo = make_pipeline(StandardScaler(), SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, decision_function_shape='ovr'))
        #
        #                 # Evaluate model using cross-validation
        #                 scores = cross_val_score(svm_modeloo, X_train, train_bins3.ravel(), cv=5)
        #                 avg_score = scores.mean()
        #
        #                 # Update best parameters and score if current model performs better
        #                 if avg_score > best_score:
        #                     best_score = avg_score
        #                     best_params = {'C': C, 'kernel': kernel, 'gamma': gamma, 'degree': degree}
        # print('SVM Best', best_score)
        # print('SVM params', best_params)
        #
        # # Perform Bayesian search for GradientBoostingClassifier
        # gbm_bayes_search = BayesSearchCV(GradientBoostingClassifier(),
        #                                  search_spaces=gbm_param_dist,
        #                                  n_iter=10, cv=5, random_state=42)
        #
        # gbm_bayes_search.fit(X_train, train_bins3.ravel())
        #
        # # Access best parameters and best scores for each model
        # print("Random Forest Best Parameters:", rf_bayes_search.best_params_)
        # print("Random Forest Best Score:", rf_bayes_search.best_score_)
        #
        #
        # print("Gradient Boosting Best Parameters:", gbm_bayes_search.best_params_)
        # print("Gradient Boosting Best Score:", gbm_bayes_search.best_score_)


        '''
        REDUNDANT - (class selection) Trying to use cohen's kappa measurement (didn't turn out useful)
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

        '''REDUNDANT -calculating entropy'''
        #REDUNDANT - for calulcating entropy for quantile binning and kmeans (same was done for uniform bins) to help choose number of classes and binning method
        # proportion1 = np.bincount(val_bins1.flatten().astype(int))/val.shape[0]
        # entropy_freq = -1*np.sum(proportion1 * np.log2(proportion1))
        # entropies_freq.append(entropy_freq)
        #
        # proportion3 = np.bincount(val_bins3.flatten().astype(int)) / val.shape[0]
        # entropy_kmeans = -1*np.sum(proportion3 * np.log2(proportion3))
        # entropies_kmeans.append(entropy_kmeans )

        '''Fitting of 3 models using only training and test after choosing number of classes,
         binning method and hyperparams'''

        popularity_train_final = np.vstack((popularity_train, popularity_val))
        splitter_kmeans_final = sklearn.preprocessing.KBinsDiscretizer(subsample=200000, n_bins=i, random_state=17,
                                                                       encode='ordinal', strategy='kmeans')
        kmeans_final = splitter_kmeans_final.fit(popularity_train_final)


        train_bins_final = kmeans_final.transform(train.loc[:, train.columns == "popularity"].values)
        val_bins_final = kmeans_final.transform(val.loc[:, val.columns == "popularity"].values)


        X_train_final = np.vstack((X_train, X_val))
        train_bins_final = np.vstack((train_bins_final, val_bins_final))


        SVM_model_tuned = make_pipeline(StandardScaler(), SVC(C=1, kernel='linear', decision_function_shape='ovr'))

        GBM_model_tuned = GradientBoostingClassifier(learning_rate = 0.020871588778809444, max_depth = 8, max_features = 'log2', min_samples_leaf = 13, min_samples_split = 13, n_estimators = 123)

        RF_model_tuned = RandomForestClassifier(bootstrap = True, max_depth =  8, max_features = None, min_samples_leaf = 4, min_samples_split = 14, n_estimators = 89)

        SVM_model_tuned.fit(X_train_final, train_bins_final.ravel())
        GBM_model_tuned.fit(X_train_final, train_bins_final.ravel())
        RF_model_tuned.fit(X_train_final, train_bins_final.ravel())

        SVMfinal = np.array(SVM_model_tuned.predict(X_test))
        GBMfinal = np.array(GBM_model_tuned.predict(X_test))
        RFfinal = np.array(RF_model_tuned.predict(X_test))

        test_bins_final = kmeans_final.transform(popularity_test)

        matsvm = sk.metrics.confusion_matrix(SVMfinal, test_bins_final.ravel())
        matgbm = sk.metrics.confusion_matrix(GBMfinal, test_bins_final.ravel())
        matrf = sk.metrics.confusion_matrix(RFfinal, test_bins_final.ravel())



        # Calculate accuracy for each confusion matrix
        accuracy_svm = calculate_accuracy(matsvm)
        accuracy_gbm = calculate_accuracy(matgbm)
        accuracy_rf = calculate_accuracy(matrf)

        importances = GBM_model_tuned.feature_importances_

        plt.bar(range(len(importances)), importances)
        plt.xlabel('Feature Number')
        plt.ylabel('Importance')
        plt.title(f"Feature importance with no. bins: {i}")
        #plt.yscale("log")
        plt.show()



        print(f"{i} bins:")
        print(" \n Confusion matrix for SVM: \n", matsvm)
        print("Accuracy for SVM:", accuracy_svm)
        print(" \n Confusion matrix for GBM: \n", matgbm)
        print("Accuracy for GBM:", accuracy_gbm)
        print(" \n Confusion matrix for Forest: \n", matrf)
        print("Accuracy for Random Forest:", accuracy_rf)
        # Print accuracy for each model

        for p in range(14):
            features, feature_names = [(p,)], ['explicit', 'Song Length', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'key','liveness','loudness','speechiness','tempo','timeSignature', 'mode', 'valence']
            partial_results = partial_dependence(GBM_model_tuned, X_test, features, method='brute')
            deciles = {0: np.linspace(0, 1, num=5)}

            display = PartialDependenceDisplay([partial_results], features=features, feature_names=feature_names,
                                               target_idx=0, deciles=deciles)
            display.plot()
            plt.savefig(f"New_{feature_names[p]}_{i}")
            plt.show()





        '''REDUNDANT - (USED FOR SELECTING NUM CLASSES Creating lists of binomially differenced accuracy Calculating datum accuracy'''
        # Binomial_TP = val.shape[0] / i
        # #random forest
        # accuracies_kmeans_R.append(np.trace(mat3) - Binomial_TP)
        #
        # #SVM
        # accuracies_kmeans_SVM.append(np.trace(SVM_mat3) - Binomial_TP)
        #
        # #GBM
        # accuracies_kmeans_GB.append(np.trace(gbm_mat3) - Binomial_TP)
        #
        # #Linear Classifier - initially used but dicarded after number of classes chosen
        # # accuracies_kmeans_lin.append(np.trace(sgd_mat3) - Binomial_TP)

        '''REDUNDANT - print sequences used to evaluate binning methods number of classes and initial model screening'''
        # print(f"{i} equal frequency bins:")
        # print('class entropy', entropy_freq)
        # print("True predictions - binomial true predictions and entropy")
        # print('Rforest', accuracies_freq_R[i-3], 'SVM', accuracies_freq_SVM[i-3], 'GBM', accuracies_freq_GB[i-3], 'Linear', accuracies_freq_lin[i-3])
        #
        # '''Analogous print sequence was used for uniform binning method'''
        # print(f"{i} kmeans bins:")
        # #print('class entropy', entropy_kmeans)
        # print('Rforest', accuracies_kmeans_R[int(i/4-1)], 'SVM', accuracies_kmeans_SVM[int(i/4-1)], 'GBM', accuracies_kmeans_GB[i/4-1], 'Linear', accuracies_kmeans_lin[i/4-1])
        # print("\n \n \n")

    '''REDUNDANT - Entropies Plot - used for evaluating binning method and number of classes'''
    # plt.plot(classes, entropies_freq, label = 'Classes with equal freq')
    # plt.plot(classes, entropies_kmeans, label = 'Classes with kmeans binning')
    # plt.legend()
    # plt.xlabel('Number of Classes')
    # plt.ylabel('Entropy')
    # plt.title('Entropies for Different Binning Methods')

    '''REDUNDANT - Differenced True Positive Graph for Equal Frequencies 
     - used for evaluating binning method and number of classes'''
    # plt.figure()
    # plt.xlabel('Number of Classes')
    # plt.ylabel('Difference between number of Correct Classifications and Random' '\n' 'Correct Guesses (n*1/(class num.), Likely Correct Binomial Guesses)')
    # plt.title('Differenced True Positive Graph for Equal Frequencies')
    # plt.plot(classes, accuracies_freq_R, label = 'Random Forest')
    # plt.plot(classes, accuracies_freq_SVM, label = 'SVM')
    # plt.plot(classes, accuracies_freq_GB, label='GBM')
    # plt.plot(classes, accuracies_freq_lin, label='Linear Classifier')
    # plt.legend()

    '''REDUNDANT - Differenced True Positive Graph for Kmeans Binning Used to select classes
     - used for evaluating binning method and number of classes'''
    # plt.figure()
    # plt.xlabel('Number of Classes')
    # plt.ylabel(
    #     'Difference between number of Correct Classifications and Random' '\n' 'Correct Guesses (n*1/(class num.), Likely Correct Binomial Guesses)')
    # plt.title('Differenced True Positive Graph for Kmeans Binning')
    # plt.plot(classes, accuracies_kmeans_R, label='Random Forest')
    # plt.plot(classes, accuracies_kmeans_SVM, label='SVM')
    # plt.plot(classes, accuracies_kmeans_GB, label='GBM')
    # plt.plot(classes, accuracies_kmeans_lin, label='Linear Classifier')
    # plt.legend()

    '''REDUNDANT - Product plot for frequency bins - used for evaluating binning method and number of classes'''
    # plt.figure()
    # plt.xlabel('Number of Classes')
    # plt.ylabel('Product of Entropy and Differenced True Classifications')
    # plt.title('Entropy x Accuracy for Equal Frequency Bins')
    # efreq = np.array(entropies_freq)
    # a_SVM_freq = np.array(accuracies_freq_SVM)
    # a_R_freq = np.array(accuracies_freq_R)
    # a_GB_freq = np.array(accuracies_freq_GB)
    # a_lin_freq = np.array(accuracies_freq_lin)
    #
    # plt.plot(classes, efreq*a_R_freq, label = 'Random Forest')
    # plt.plot(classes, efreq*a_SVM_freq, label = 'SVM')
    # plt.plot(classes, efreq * a_GB_freq, label='GBM')
    # plt.plot(classes, efreq * a_lin_freq, label='Linear Classifier')
    # plt.legend()

    '''REDUNDANT - Product plot for kmeans bins - used for evaluating binning method and number of classes'''
    # plt.figure()
    # plt.xlabel('Number of Classes')
    # plt.ylabel('Product of Entropy and Differenced True Classifications')
    # plt.title('Entropy x Accuracy for Kmeans Bins')
    # ek = np.array(entropies_kmeans)
    # a_SVM_kmeans = np.array(accuracies_kmeans_SVM)
    # a_R_kmeans = np.array(accuracies_kmeans_R)
    # a_GB_kmeans = np.array(accuracies_kmeans_GB)
    # a_lin_kmeans = np.array(accuracies_kmeans_lin)
    #
    # plt.plot(classes, ek * a_R_kmeans, label='Random Forest')
    # plt.plot(classes, ek * a_SVM_kmeans, label='SVM')
    # plt.plot(classes, ek * a_GB_kmeans, label='GBM')
    # plt.plot(classes, ek * a_lin_kmeans, label='Linear Classifier')
    # plt.legend()
    #
    # plt.show()

class_finder(train, val, test)



