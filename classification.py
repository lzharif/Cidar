import glob
from os import listdir
from os.path import isfile, join

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
    scal = './Model/scale.sav'
    proc_lda = './Model/lda.sav'
    proc_pca = './Model/pca.sav'

    model_dt = './Model/DT_noproc.sav'
    model_knn = './Model/kNN_noproc.sav'
    model_nb = './Model/NB_lda.sav'
    model_rf = './Model/RF_noproc.sav'
    model_svm = './Model/SVM_pca.sav'

    def __init__(self):
        pass

    def klasifikasiCitraBanyak(self, folder, method):
        self.folder = folder
        files = self.listFiles(folder)
        scale, proc, klas = self.loadModel(method)
        hasil = []
        for fil in files:
            fe = FeatureExtraction(fil)
            fitur = fe.ekstraksifitur()
            jenis = self.klaf(scale, fitur, proc, klas)
            hasil.append(jenis)
        
        return hasil

    def listFiles(self, folder):
        types = ('/*.jpg', '/*.png')
        files = []
        for tipe in types:
            a = folder + tipe
            files.extend(glob.glob(folder + tipe))
        return files

    def loadModel(self, method):
        scale = pickle.load(open(self.scal, 'rb'))
        proc = None
        klas = None
        if method == 'Decision Tree':
            klas = pickle.load(open(self.model_dt, 'rb'))
        elif method == 'kNN':
            klas = pickle.load(open(self.model_knn, 'rb'))
        elif method == 'Neural Network':
            # TODO Buat fungsi buka model Neural Network
            a = 1
        elif method == 'Naive Bayes':
            proc = pickle.load(open(self.proc_lda, 'rb'))
            klas = pickle.load(open(self.model_nb, 'rb'))
        elif method == 'Random Forest':
            klas = pickle.load(open(self.model_rf, 'rb'))
        else:
            proc = pickle.load(open(self.proc_pca, 'rb'))
            klas = pickle.load(open(self.model_svm, 'rb'))
        return scale, proc, klas

    def klaf(self, scale, fitur, proc, klas):
        fit = scale.transform(fitur)
        hasil = 0
        if proc == None:
            hasil = klas.predict(fit)
        else:
            fit = proc.transform(fit)
            hasil = klas.predict(fit)
        return hasil[0]