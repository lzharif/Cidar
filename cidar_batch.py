from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import glob
import os

from feature_extraction import FeatureExtraction
# from classification import Classify

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

        # fileMenu.add_command(label="Buka Satuan", underline=0, command=lambda: self.tangkapCitraSatuan())
        # fileMenu.add_command(label="Buka Banyak", underline=0, command=lambda: self.ambilFolder())
        fileMenu.add_command(label="Simpan Data", underline=0, command=lambda: self.simpanData())

        fileMenu.add_separator()

        fileMenu.add_command(label="Keluar", underline=0, command=self.onExit)
        menubar.add_cascade(label="Berkas", underline=0, menu=fileMenu)


        labelTahap1 = Label(self.parent, text = "Tahap 1: Pilih folder citra", justify = LEFT).grid(sticky = W, columnspan = 2)

        labelFolder = Label(self.parent, text = "Belum pilih folder")
        buttonPilihFolder = Button(self.parent, text = "Pilih Folder", command = lambda: self.ambilFolder(labelFolder)).grid(sticky = W, row = 1, column = 0)
        labelFolder.grid(sticky = W, row = 1, column = 1)
        labelSep = Label(self.parent, text = "").grid(sticky = W, columnspan = 2, row = 2)
        labelTahap2 = Label(self.parent, text = "Tahap 2: Pilih metode kecerdasan buatan").grid(sticky = W, columnspan = 2, row = 3)

        # Membuat Dropdown
        self.kecerdasan = 'Decision Tree'
        mainframe = Frame(self.parent)
        mainframe.grid(column=0,row=4, sticky=(N,W,E,S) )
        mainframe.columnconfigure(0, weight = 1)
        mainframe.rowconfigure(0, weight = 1)
        
        tkvar = StringVar(self.parent)
        
        choices = { 'Decision Tree','kNN','Naive Bayes','Neural Network','Random Forest', 'SVM'}
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
        labelHasil = Label(self.parent, text = "Hasil").grid(row = 9, column = 0)
        labelJumlah = Label(self.parent, text = "Jumlah Citra:").grid(sticky = W, row = 10, columnspan = 2, column = 0)
        self.labelBasofil = Label(self.parent, text = "Jumlah Basofil:").grid(sticky = W, row = 11, column = 0, columnspan = 2)
        labelEosinofil = Label(self.parent, text = "Jumlah Eosinofil:").grid(sticky = W, row = 12, column = 0, columnspan = 2)
        labelLimfosit = Label(self.parent, text = "Jumlah Limfosit:").grid(sticky = W, row = 13, column = 0, columnspan = 2)
        labelMonosit = Label(self.parent, text = "Jumlah Monosit:").grid(sticky = W, row = 14, column = 0, columnspan = 2)
        labelNetrofil = Label(self.parent, text = "Jumlah Netrofil:").grid(sticky = W, row = 15, column = 0, columnspan = 2)

        labelSep2 = Label(self.parent, text = "").grid(sticky = W, columnspan = 2, row = 5)
        labelTahap3 = Label(self.parent, text = "Tahap 3: Tekan jalankan").grid(sticky = E, row = 6)
        buttonJalan = Button(self.parent, text = "Jalankan", command = lambda: self.olahBanyak(labelJumlah, labelBasofil, labelEosinofil, labelLimfosit, labelMonosit, labelNetrofil)).grid(sticky = W, row = 7, column = 0)


    def ambilFolder(self, label):
        dlg = filedialog.askdirectory()
        if dlg != '':
            self.folderCitra = os.path.dirname(os.path.abspath(dlg))
            label.config(text = str(self.folderCitra))
            pass

    def olahBanyak(self, lblJumlah, lblB, lblE, lblL, lblM, lblN):
        if self.folderCitra != '':
            #TODO Buat fungsi Klasifikasi
            berkas = self.fol

    def tangkapCitraSatuan(self):
        ftypes = [('Portable Network Graphics', '*.png'), ('JPEG', '*.jpg')]
        dlg = filedialog.Open(self, filetypes=ftypes)
        fl = dlg.show()

        if fl != '':
            # self.img = Image.open(fl)
            # f = ImageTk.PhotoImage(self.img)
            citra = cv2.imread(fl,cv2.IMREAD_COLOR)
            fres = self.resizeImage(citra)

            # if APA_CITRA_PENUH == True:
            #     jumlahWBC, hasilResize = prosesCitraPenuh(f)
            
            fe = FeatureExtraction(citra)
            fitur = fe.ekstraksifitur()
            print(fitur)

            # Agar bisa dibuka secara benar oleh Tk
            fres = cv2.cvtColor(fres, cv2.COLOR_BGR2RGB)
            fres = Image.fromarray(fres)
            fres = ImageTk.PhotoImage(fres)

            # label = Label(self, height="480", width="640", image=fres)  # img nanti di resize dulu dari hasil TensorBox
            # label.image = fres
            # label.pack(side="left", expand="no")
    
    def prosesCitraBanyak(self, dirCitra):
        pass

    def simpanData(self):
        pass

    def onExit(self):
        self.quit()

    def resizeImage(self, f):
        return cv2.resize(f, (640, 480))


def main():
    # Inisialisasi TensorBox dan TensorFlow
    global seGmentasi
    global seKlasifikasi
    root = Tk()
    app = GUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()