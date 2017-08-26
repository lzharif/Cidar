# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 17:15:15 2017

@author: Visual.Sensor
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
# Applying Grid Search to find the best hyperparameter
from sklearn.model_selection import GridSearchCV

# Name
filename_noproc = 'NB_noproc.sav'
filename_pca = 'NB_pca.sav'
filename_lda = 'NB_lda.sav'
filename_kpca = 'NB_kpca.sav'
filename_scale = 'scale.sav'
filename_dr_pca = 'pca.sav'
filename_dr_lda = 'lda.sav'
filename_dr_kpca = 'kpca.sav'
filename_res_noproc = 'NB_res_noproc.txt'
filename_res_pca = 'NB_res_pca.txt'
filename_res_lda = 'NB_res_lda.txt'
filename_res_kpca = 'NB_res_kpca.txt'

# Grid Searching with Parallel Computing
def cariGrid(clsf, preproc, xtr, ytr, xte, yte, accu, std, test_accu):
    parameters = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 4, 10, 20, 30],
              "max_depth": [None, 2, 5, 10, 15, 20],
              "min_samples_leaf": [1, 5, 10, 20],
              "max_leaf_nodes": [None, 5, 10, 15, 20]
              }
    grid_search = GridSearchCV(estimator=clsf,
                               param_grid=parameters,
                               scoring='accuracy',
                               cv=10,
                               n_jobs=-1,
                               verbose=0)
    grid_search = grid_search.fit(xtr, ytr)
    best_accuracy = grid_search.best_score_
    best_index = grid_search.best_index_
    best_std = grid_search.cv_results_['std_test_score'][best_index]
    best_parameters = grid_search.best_params_

    clsf = GaussianNB(**best_parameters).fit(xtr, ytr)

    # Calculate test accuracy with optimized training
    test_optimized = grid_search.score(xte, yte)

    if preproc == 'noproc':
        pickle.dump(clsf, open(filename_noproc, 'wb'))
        with open(filename_res_noproc, "w") as text_file:
            text_file.write("%f %f %f %f %f %f %s" % (accu, std, test_accu, best_accuracy, best_std, test_optimized, best_parameters))
    elif preproc == 'pca':
        pickle.dump(clsf, open(filename_pca, 'wb'))
        with open(filename_res_pca, "w") as text_file:
            text_file.write("%f %f %f %f %f %f %s" % (accu, std, test_accu, best_accuracy, best_std, test_optimized, best_parameters))
    elif preproc == 'lda':
        pickle.dump(clsf, open(filename_lda, 'wb'))
        with open(filename_res_lda, "w") as text_file:
            text_file.write("%f %f %f %f %f %f %s" % (accu, std, test_accu, best_accuracy, best_std, test_optimized, best_parameters))
    else:
        pickle.dump(clsf, open(filename_kpca, 'wb'))
        with open(filename_res_kpca, "w") as text_file:
            text_file.write("%f %f %f %f %f %f %s" % (accu, std, test_accu, best_accuracy, best_std, test_optimized, best_parameters))
    
    print(best_accuracy)
    print(best_std)
    print(best_parameters)
    print(test_optimized)

# Importing the dataset
dataset = pd.read_csv('100_auto.csv', sep=';')
X = dataset.iloc[:, 0:29].values
y = dataset.iloc[:, 29].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
pickle.dump(sc, open(filename_scale, 'wb'))

preprocess = ['noproc', 'pca', 'lda', 'kpca']
for i in preprocess:
    if i == 'noproc':
        pass
    elif i == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components = 10)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        explained_variance = pca.explained_variance_ratio_
        pickle.dump(pca, open(filename_dr_pca, 'wb'))
    elif i == 'lda':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        lda = LDA(n_components= 5, )
        X_train = lda.fit_transform(X_train, y_train)
        X_test = lda.transform(X_test)
        pickle.dump(lda, open(filename_dr_lda, 'wb'))
    else:
        from sklearn.decomposition import KernelPCA
        kpca = KernelPCA(n_components= 10, kernel='rbf')
        X_train = kpca.fit_transform(X_train)
        X_test = kpca.transform(X_test)
        pickle.dump(kpca, open(filename_dr_kpca, 'wb'))

    # Fitting classifier to the Training set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Applying k-Fold Cross Validation
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
    avg_accuracy = accuracies.mean()
    std_accuracy = accuracies.std()
    print('Akurasi: ', avg_accuracy)
    print('SD: ', std_accuracy)

    # Test unoptimized performance
    # Predicting the Test set results
    test_accuracy = classifier.score(X_test, y_test)
    print('Akurasi Tes:', test_accuracy)

    # Making the Confusion Matrix
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    if i == 'noproc':
        pickle.dump(classifier, open(filename_noproc, 'wb'))
        with open(filename_res_noproc, "w") as text_file:
            text_file.write("%f %f %f" % (avg_accuracy, std_accuracy, test_accuracy))
    elif i == 'pca':
        pickle.dump(classifier, open(filename_pca, 'wb'))
        with open(filename_res_pca, "w") as text_file:
            text_file.write("%f %f %f" % (avg_accuracy, std_accuracy, test_accuracy))
    elif i == 'lda':
        pickle.dump(classifier, open(filename_lda, 'wb'))
        with open(filename_res_lda, "w") as text_file:
            text_file.write("%f %f %f" % (avg_accuracy, std_accuracy, test_accuracy))
    else:
        pickle.dump(classifier, open(filename_kpca, 'wb'))
        with open(filename_res_kpca, "w") as text_file:
            text_file.write("%f %f %f" % (avg_accuracy, std_accuracy, test_accuracy))

    # if __name__ == '__main__':
    #     cariGrid(classifier, i, X_train, y_train, X_test, y_test, avg_accuracy, std_accuracy, test_accuracy)