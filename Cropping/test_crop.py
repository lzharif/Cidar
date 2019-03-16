import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt

def convToHSV(img):
    imgc = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(imgc)
    return s

def filterImg(img):
    blur = cv2.GaussianBlur(img,(7,7),0)
    ret, th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th

def aturNama(nama, idx):
    #TODO Buat Regular Expression untuk ambil nama
    namaFix = nama
    kata = nama.split('.')
    if (idx != 0):
        namaFix = kata[0] + '_' + str(idx) + '.' + kata[1]
    return namaFix

    
def simpanKontur(img, cnt, nama, idx):
    offset = 75
    x_awal = 0
    y_awal = 0
    x_akhir = 0
    y_akhir = 0
    img_height,img_width = img.shape[:2]

    x,y,w,h = cv2.boundingRect(cnt)
    if (y >= offset):
        y_awal = y - offset
    else:
        y_awal = y

    if (x >= offset):
        x_awal = x - offset
    else:
        x_awal = x

    x_akhir = x + w + offset
    y_akhir = y + h + offset
    if (x_akhir > img_width):
        x_akhir = img_width
    if (y_akhir > img_height):
        y_akhir = img_height

    hasil = img[y_awal:y_akhir, x_awal:x_akhir]
    namafile = aturNama(nama, idx)
    cv2.imwrite(namafile, hasil)

def kontur(img, imageAsli, nama):
    threshArea = 9000
    gmb, cont, hier = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(img.shape, np.uint8)
    idx = 0
    for cnt in cont:
        area = cv2.contourArea(cnt)
        if area > threshArea and area < img.shape[0] * img.shape[1]:
            simpanKontur(imageAsli, cnt, nama, idx)
            cv2.drawContours(mask, [cnt], -1, (255,255,255,255), -1)
            idx = idx + 1
    
    hasil = cv2.bitwise_and(imageAsli, imageAsli, mask= mask)
    return hasil

def listFiles(folder):
    types = ('*.jpg', '*.png')
    nama = []
    for tipe in types:
        nama.extend(glob.glob(tipe))
    return nama

cwd = os.getcwd()
gambar2 = listFiles(cwd)
print(gambar2)
for gmb in gambar2:
    img = cv2.imread(gmb)
    img2 = convToHSV(img)
    img2 = filterImg(img2)
    hasil = kontur(img2, img, gmb)
# cv2.imwrite('abal1.jpg', hasil)

# i = 0
# for fil in files:
#     img = cv2.imread(fil)
#     s = convToHSV(img)
#     cv2.imwrite('abal1.jpg', s)
#     i = i + 1

cv2.waitKey(0)