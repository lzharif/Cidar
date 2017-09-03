import cv2
import glob
import os
import numpy as np

from PIL import Image, ImageTk
from tkinter import *
from tkinter import filedialog

from classification import Classify
from feature_extraction import FeatureExtraction
from helper import Helper

# Variabel Konstan
JENIS_SEL_DARAH_PUTIH = 5
APA_CITRA_PENUH = False

class GUI(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)

        self.parent = parent

        self.initUI()

    # Fungsi untuk Inisialisasi tampilan
    def initUI(self):
        self.parent.title("Cidar")

        menubar = Menu(self.parent)
        self.parent.config(menu=menubar)

        fileMenu = Menu(menubar)

        fileMenu.add_command(label="Ekstraksi Banyak", underline=0, command=lambda: self.simpanData())

        fileMenu.add_separator()

        fileMenu.add_command(label="Keluar", underline=0, command=self.onExit)
        menubar.add_cascade(label="Berkas", underline=0, menu=fileMenu)


        labelTahap1 = Label(self.parent, text = "Tahap 1: Pilih folder citra", justify = LEFT).grid(sticky = W, columnspan = 2)

        labelFolder = Label(self.parent, text = "Belum pilih folder")
        buttonPilihFolder = Button(self.parent, text = "Pilih Folder", command = lambda: self.ambilFolder(labelFolder)).grid(sticky = W, row = 1, column = 0)
        buttonPilihFolder = Button(self.parent, text = "Ekstraksi Banyak", command = lambda: self.ekstrakBanyak()).grid(sticky = W, row = 2, column = 0)
        labelFolder.grid(sticky = W, row = 1, column = 1, columnspan = 5)
        labelSep = Label(self.parent, text = "").grid(sticky = W, columnspan = 2, row = 2)
        labelTahap2 = Label(self.parent, text = "Tahap 2: Pilih metode kecerdasan buatan").grid(sticky = W, columnspan = 2, row = 3)

        # Membuat Dropdown
        self.kecerdasan = 'Decision Tree'
        mainframe = Frame(self.parent)
        mainframe.grid(column=0,row=4, sticky=(N,W,E,S) )
        mainframe.columnconfigure(0, weight = 1)
        mainframe.rowconfigure(0, weight = 1)
        
        tkvar = StringVar(self.parent)
        
        choices = { 'Decision Tree','kNN','Naive Bayes', 'Neural Network','Random Forest', 'SVM'}
        tkvar.set('Decision Tree') # set the default option
        
        popupMenu = OptionMenu(mainframe, tkvar, *choices)
        popupMenu.grid(sticky = W, row = 4, column =0)
 
        # on change dropdown value
        def change_dropdown(*args):
            self.kecerdasan = tkvar.get()
            print( tkvar.get() )
        
        # link function to change dropdown
        tkvar.trace('w', change_dropdown)

        # Tempat nampilin hasil
        labelPemisah = Label(self.parent, text = "-----------------------------------------").grid(sticky = W, columnspan = 3, row = 8, column = 0, pady = 5)
        labelHasil = Label(self.parent, text = "Hasil")
        labelJumlah = Label(self.parent, text = "Jumlah Citra:")
        labelBasofil = Label(self.parent, text = "Jumlah Basofil:")
        labelEosinofil = Label(self.parent, text = "Jumlah Eosinofil:")
        labelLimfosit = Label(self.parent, text = "Jumlah Limfosit:")
        labelMonosit = Label(self.parent, text = "Jumlah Monosit:")
        labelNetrofil = Label(self.parent, text = "Jumlah Netrofil:")
        labelPhoto = Label(self.parent)

        labelHasil.grid(row = 9, column = 0)
        labelJumlah.grid(sticky = W, row = 10, columnspan = 2, column = 0)
        labelBasofil.grid(sticky = W, row = 11, column = 0, columnspan = 2)
        labelEosinofil.grid(sticky = W, row = 12, column = 0, columnspan = 2)
        labelLimfosit.grid(sticky = W, row = 13, column = 0, columnspan = 2)
        labelMonosit.grid(sticky = W, row = 14, column = 0, columnspan = 2)
        labelNetrofil.grid(sticky = W, row = 15, column = 0, columnspan = 2)
        labelPhoto.grid(sticky = W, row = 10, column = 2, columnspan = 2)

        labelSep2 = Label(self.parent, text = "").grid(sticky = W, columnspan = 2, row = 5)
        labelTahap3 = Label(self.parent, text = "Tahap 3: Tekan jalankan").grid(sticky = E, row = 6)
        buttonJalan = Button(self.parent, text = "Jalankan", command = lambda: self.olahBanyak(labelJumlah, labelBasofil, labelEosinofil, labelLimfosit, labelMonosit, labelNetrofil, labelPhoto)).grid(sticky = W, row = 7, column = 0)

    def ambilFolder(self, label):
        dlg = filedialog.askdirectory()
        print(dlg)
        if dlg != '':
            self.folderCitra = dlg
            label.config(text = str(self.folderCitra))
            pass

    def olahBanyak(self, lblJumlah, lblB, lblE, lblL, lblM, lblN, lblPhoto):
        if self.folderCitra != '':
            berkas = self.folderCitra
            clf = Classify()
            hasil_batch = clf.klasifikasiCitraBanyak(berkas, self.kecerdasan)
            bas = (hasil_batch == 0).sum()
            eos = (hasil_batch == 1).sum()
            lim = (hasil_batch == 2).sum()
            mon = (hasil_batch == 3).sum()
            net = (hasil_batch == 4).sum()
            banyak = len(hasil_batch)

            rerBas = float("{0:.2f}".format(bas*100/banyak))
            rerEos = float("{0:.2f}".format(eos*100/banyak))
            rerLim = float("{0:.2f}".format(lim*100/banyak))
            rerMon = float("{0:.2f}".format(mon*100/banyak))
            rerNet = float("{0:.2f}".format(net*100/banyak))
            lblJumlah.config(text = "Jumlah Citra: " + str(banyak))
            lblB.config(text = "Jumlah Basofil: " + str(bas) + " -> " + str(rerBas))
            lblE.config(text = "Jumlah Eosinofil: " + str(eos) + " -> " + str(rerEos))
            lblL.config(text = "Jumlah Limfosit: " + str(lim) + " -> " + str(rerLim))
            lblM.config(text = "Jumlah Monosit: " + str(mon) + " -> " + str(rerMon))
            lblN.config(text = "Jumlah Netrofil: " + str(net) + " -> " + str(rerNet))

            # Baca gambar confusion matrix
            clf.ambilConfusionMatrix(self.folderCitra, hasil_batch)
            image_conf = Image.open(self.folderCitra + '/confusion_matrix.png')
            if image_conf != None:
                lblPhoto.config(image = image_conf)
            
    def tangkapCitraSatuan(self):
        ftypes = [('Portable Network Graphics', '*.png'), ('JPEG', '*.jpg')]
        dlg = filedialog.Open(self, filetypes=ftypes)
        fl = dlg.show()

        # if fl != '':
        #     # self.img = Image.open(fl)
        #     # f = ImageTk.PhotoImage(self.img)
        #     citra = cv2.imread(fl,cv2.IMREAD_COLOR)
        #     fres = self.resizeImage(citra)

        #     # if APA_CITRA_PENUH == True:
        #     #     jumlahWBC, hasilResize = prosesCitraPenuh(f)
            
        #     fe = FeatureExtraction(citra)
        #     fitur = fe.ekstraksifitur()
        #     print(fitur)

        #     # Agar bisa dibuka secara benar oleh Tk
        #     fres = cv2.cvtColor(fres, cv2.COLOR_BGR2RGB)
        #     fres = Image.fromarray(fres)
        #     fres = ImageTk.PhotoImage(fres)

            # label = Label(self, height="480", width="640", image=fres)  # img nanti di resize dulu dari hasil TensorBox
            # label.image = fres
            # label.pack(side="left", expand="no")
    
    def ekstrakBanyak(self):
        dlg = filedialog.askdirectory()
        print(dlg)
        if dlg != '':
            self.folderCitra = dlg
            fe = FeatureExtraction()
            fitur2 = fe.ekstrakFiturBanyak(self.folderCitra)
            
            pass

    def simpanData(self):
        pass

    def onExit(self):
        self.quit()

    def resizeImage(self, f):
        return cv2.resize(f, (640, 480))


def main():
    global seGmentasi
    global seKlasifikasi
    root = Tk()
    app = GUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()