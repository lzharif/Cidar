import itertools
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from feature_extraction import FeatureExtraction
from helper import Helper

# Classify adalah kelas yang melaksanakan fungsi klasifikasi
class Classify:
    # Inisialisasi model kecerdasan buatan yang telah dilatih sebelumnya
    # Singkatan praproses:
    # SCAL = Standard Scaler
    # LDA = Linear Discriminant Analysis
    # PCA = Principal Component Analysis

    # Singkatan algoritma kecerdasan buatan:
    # DT = Decision Tree
    # KNN = K-Nearest Neighbour
    # NB = Naive Bayes
    # RF = Random Forest
    # SVM = Support Vector Machine

    model_fol = './Model'
    scal = './Model/scale.sav'
    proc_lda = './Model/lda.sav'
    proc_pca = './Model/pca.sav'

    model_dt = './Model/DT_noproc.sav'
    model_knn = './Model/kNN_noproc.sav'
    model_nb = './Model/NB_lda.sav' # awalnya './Model/NB_pca.sav'
    model_rf = './Model/RF_noproc.sav'
    model_svm = './Model/SVM_noproc.sav'
    model_svm_giemsa = './Model/SVM_noproc_g.sav'
    model_svm_wright = './Model/SVM_noproc_w.sav'

    def __init__(self):
        pass

    # Fungsi untuk mengklasifikasi banyak citra sekaligus
    # Citra yang dapat diklasifikasi hanya yang sudah di crop saja
    def klasifikasiCitraBanyak(self, folder, method):
        self.folder = folder
        helper = Helper()
        files = helper.listFiles(folder)
        scale, proc, klas = self.loadModel(method)
        fitur_banyak = []
        hasil = []
        for fil in files:
            fe = FeatureExtraction()
            fitur_banyak.append(fe.ekstraksifitur(fil))
        hasil = self.klaf(scale, fitur_banyak, proc, method, klas)
        
        return hasil

    # Fungsi untuk mengklasifikasi data citra sel darah putih berformat teks
    # Berkas teks disimpan dengan nama "Hasil Ekstraksi.txt"
    # Formatnya adalah desimal menggunakan titik (.) dan pemisah koma (,), tidak ada headernya
    # Contoh:
    # 2034,20.4,133,1, ... , 15.45
    def klasifikasiTeks(self, folder, method):
        self.folder = folder
        berkas_teks = open(folder + "/Hasil Ekstraksi.txt", "r")
        fitur_banyak = []
        hasil = []
        if berkas_teks != None:
            fitur_banyak = np.loadtxt(berkas_teks,delimiter=',')
            scale, proc, klas = self.loadModel(method)
            hasil = self.klaf(scale, fitur_banyak, proc, method, klas)
        
            return hasil
        
    # Fungsi untuk menghitung dan menampilkan confusion matrix
    # Fungsi ini membutuhkan "truth.txt" yang berisi kelas citra
    # Kelas citra basofil, eosinofil, limfosit, monosit, neutrofil, dan stab berurut dari 0-5
    # Hanya pada citra dengan pewarnaan giemsa yang tidak mencantumkan basofil, sehingga urutan kelas 0-4
    # Contoh:
    # 5,1,0,0,4
    #
    # Tandanya baris pertama adalah stab, kedua eosinofil, dan kelima adalah neutrofil
    def ambilConfusionMatrix(self, folder, prediksi):
        self.folder = folder
        truth_file = open(folder + "/truth.txt", "r")
        if truth_file != None:
            y_true = truth_file.read().split(",")
            y_true_val = list(map(int, y_true))
            conf = confusion_matrix(y_true_val, prediksi)

            plt.figure()
            classes = None
            apakahWright = False
            if apakahWright == True:
                self.plot_confusion_matrix(conf, classes=[0, 1, 2, 3, 4, 5], title='Confusion matrix, without normalization')
            else:
                self.plot_confusion_matrix(conf, classes=[0, 1, 2, 3, 4], title='Confusion matrix, without normalization')
            plt.show()

    # Fungsi bantuan untuk membuat grafik confusion matrix
    # Digunakan pada fungsi ambilConfusionMatrix()
    def plot_confusion_matrix(self, cm, classes,
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

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
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

    # Fungsi untuk memuat model terpilih yang telah tersimpan sebelumnya
    # Semua model harus disimpan di folder ./Model
    def loadModel(self, method):
        scale = pickle.load(open(self.scal, 'rb'))
        proc = None
        klas = None
        if method == 'Decision Tree':
            klas = pickle.load(open(self.model_dt, 'rb'))
        elif method == 'kNN':
            klas = pickle.load(open(self.model_knn, 'rb'))
        elif method == 'Neural Network':
            dimension = 29
            hidden_layers = [100, 100]
            feature_columns = [tf.contrib.layers.real_valued_column("", dimension=dimension)]  # Banyaknya fitur
            klas = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=hidden_layers,
                                                n_classes=5,  # banyak kelas, dalam darah ada 5
                                                model_dir= self.model_fol)
        elif method == 'Naive Bayes':
            proc = pickle.load(open(self.proc_lda, 'rb')) # awalnya self.proc_pca
            klas = pickle.load(open(self.model_nb, 'rb'))
        elif method == 'Random Forest':
            klas = pickle.load(open(self.model_rf, 'rb'))
        elif method == 'SVM Giemsa':
            klas = pickle.load(open(self.model_svm_giemsa, 'rb'))
        elif method == 'SVM Wright':
            klas = pickle.load(open(self.model_svm_wright, 'rb'))
        else:
            klas = pickle.load(open(self.model_svm, 'rb'))
        return scale, proc, klas

    # Fungsi utama dari kelas Classify()
    # Berisi urutan eksekusi fungsi pada kelas ini
    def klaf(self, scale, fitur, proc, method, klas):
        fitur_scaled = []
        fitur_fixed = []
        hasil = []
        if method == 'Neural Network':
            fitur_fixed = np.array(fitur, dtype=float)
            hasil = np.asarray(list(klas.predict(fitur_fixed, as_iterable = True)))
        else:
            if proc == None:
                fitur_scaled = scale.transform(fitur)
                hasil = klas.predict(fitur_scaled)
            else:
                fitur_scaled = scale.transform(fitur)
                fitur_fixed = proc.transform(fitur_scaled)
                hasil = klas.predict(fitur_fixed)
        return hasil