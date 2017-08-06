#!/usr/bin/python

from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import glob
import numpy as np
import tensorflow as tf

from train import build_forward
from utils import train_utils
from utils.annolist import AnnotationLib as al
from utils.stitch_wrapper import stitch_rects
from utils.train_utils import add_rectangles
from utils.rect import Rect
from utils.stitch_wrapper import stitch_rects
from evaluate import add_rectangles

# Variabel Konstan
JENIS_SEL_DARAH_PUTIH = 5

def resizeImage(f):
    return cv2.resize(f, (640, 480))


def segmentasiCitra(citraKecil, sess):

    # new_img, rects = add_rectangles(H, [img], np_pred_confidences, np_pred_boxes,
    #                                 use_stitching=True, rnn_len=H['rnn_len'], min_conf=0.7,
    #                                 show_suppressed=False)
    # Olah menggunakan TensorBox
    return spasial, cWBC


def klasifikasiWBC(citra, dataSpasialWBC):
    return data


def prosesCitra(citra):
    citraKecil = cv2.resize(citra, (640, 480))
    dataSpasialWBC, citraWBC = segmentasiCitra(citraKecil)
    data = klasifikasiWBC(citra, dataSpasialWBC)
    return data, citraKecil


def prosesCitraBanyak(alamatFolder):
    listCitra = glob.glob(alamatFolder + "*.png")
    segmenSaver = tf.train.Saver()
    with tf.Session() as segmenSess:
        segmenSess.run(tf.initialize_all_variables())
        segmenSaver.restore(segmenSess, 'segmensave.ckpt') # TODO edit ini sesuai nama file
        for indexCitra in range(0, listCitra.size):
            citra = cv2.imread(listCitra[indexCitra])
            citraKecil = cv2.resize(citra, (640, 480))
            dataSpasialWBC, citraWBC = segmentasiCitra(citraKecil, segmenSess)
            data = klasifikasiWBC(citra, dataSpasialWBC)

    return data, citraKecil


class GUI(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)

        self.dataJumlahWBC = np.empty(JENIS_SEL_DARAH_PUTIH)
        self.parent = parent

        self.initUI()

    def initUI(self):
        self.parent.title("Cidar")

        menubar = Menu(self.parent)
        self.parent.config(menu=menubar)

        fileMenu = Menu(menubar)

        fileMenu.add_command(label="Buka Satuan", underline=0, command=self.tangkapCitraSatuan())
        fileMenu.add_command(label="Buka Banyak", underline=0, command=self.tangkapCitraBanyak())
        fileMenu.add_command(label="Simpan Data", underline=0, command=self.simpanData())

        fileMenu.add_separator()

        fileMenu.add_command(label="Keluar", underline=0, command=self.onExit)
        menubar.add_cascade(label="Berkas", underline=0, menu=fileMenu)

        self.pack()

    def tangkapCitraSatuan(self):
        ftypes = [('Portable Network Graphics', '*.png'), ('JPEG', '*.jpg')]
        dlg = filedialog.Open(self, filetypes=ftypes)
        fl = dlg.show()

        if fl != '':
            self.img = Image.open(fl)
            f = ImageTk.PhotoImage(self.img)

            jumlahWBC, hasilResize = prosesCitra(f)

            self.dataJumlahWBC = self.dataJumlahWBC + jumlahWBC  # Penambahan untuk update data
            label = Label(self, height="480", width="640", image=hasilResize)  # img nanti di resize dulu dari hasil TensorBox
            label.image = hasilResize
            label.pack(side="left", expand="no")

    def tangkapCitraBanyak(self):
        dlg = filedialog.askdirectory()
        dirC = dlg.show()

        if dirC != '':
            self.dataJumlahWBC, citraKotakan = prosesCitraBanyak(dirC)


    def simpanData(self):
        pass

    def onExit(self):
        self.quit()


def main():
    # Inisialisasi TensorBox dan TensorFlow
    global seGmentasi
    global seKlasifikasi
    root = Tk()
    root.geometry("1024x768+0+0")
    app = GUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()

# import os
# import cv2
# import math
# import numpy as np
# from tkinter import *
# from random import randint
#
# # logo = PhotoImage(file='contohbagus.gif')
#
# # GUI Cidar
# top = Tk()
# # top.iconbitmap(lokasi + '\logo.ico') #Set logo
# frame = Frame(top)
# frame.pack()
#
# C = Canvas(top, bg="white", height=600, width=800)
#
#
# def tangkapCitraSatuan():
#     pass
#
#
# def tangkapCitraBanyak():
#     pass
#
#
# BtnMulaiSatuan = Button(top, text='Ambil Citra', command=lambda: tangkapCitraSatuan())
# BtnMulaiBanyak = Button(top, text='Ambil Banyak', command=lambda: tangkapCitraBanyak())
# BtnKeluar = Button(top, text='Keluar', command=lambda: top.quit())
#
# C.pack(side=TOP)
# BtnMulaiSatuan.pack(padx=5, side=LEFT)
# BtnMulaiBanyak.pack(padx=5, side=LEFT)
# BtnKeluar.pack(padx=5, side=LEFT)
#
# # Fungsi mulai di sini
#
# top.title('Cidar')
# # top.tk.call('wm', 'iconphoto', top._w, logo)
# top.mainloop()
