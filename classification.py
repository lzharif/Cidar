from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from feature_extraction import FeatureExtraction

import pickle

class Classify:
    proc_pca = 'pca.sav'
    proc_lda = 'lda.sav'

    model_dt = 'DT_noproc.sav'
    model_knn = 'kNN_pca.sav'
    model_nb = 'NB_lda.sav'
    model_rf = 'RF_noproc.sav'
    model_svm = 'SVM_pca.sav'

    def __init__(self):
        pass

    def klasifikasiCitraBanyak(self, folder, method):
        self.folder = folder
        hasil = []
        for fil in folder:
            fe = FeatureExtraction(fil)
            fitur = fe.ekstraksifitur()
            jenis = self.klaf(fitur, method)
            hasil.append(jenis)
        
        return hasil

    def klaf(self, fitur, method):
        fit = fitur
        hasil = 0
        if method == 'Decision Tree':
            klas = pickle.load(open(self.model_dt, 'rb'))
            hasil = klas.predict(fit)
        elif method == 'kNN':
            proc = pickle.load(open(self.proc_pca, 'rb'))
            klas = pickle.load(open(self.model_knn, 'rb'))
            fit = proc.transform(fit)
            hasil = klas.predict(fit)
        elif method == 'Naive Bayes':
            proc = pickle.load(open(self.proc_lda, 'rb'))
            klas = pickle.load(open(self.model_nb, 'rb'))
            fit = proc.transform(fit)
            hasil = klas.predict(fit)
        elif method == 'Random Forest':
            klas = pickle.load(open(self.model_rf, 'rb'))
            hasil = klas.predict(fit)
        else:
            proc = pickle.load(open(self.proc_pca, 'rb'))
            klas = pickle.load(open(self.model_svm, 'rb'))
            fit = proc.transform(fit)
            hasil = klas.predict(fit)

        return hasil




