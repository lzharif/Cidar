import cv2
import numpy as np
import math


class FeatureExtraction:
    citra = None

    def __init__(self, c):
        # Masukkan citra ke dalam class
        self.citra = c
        # Menandakan Fitur Ekstraksi Sedang Berjalan
        print('Ekstraksi fitur citra sedang dilakukan...')

    def ekstraksifitur(self):
        citra = self.citra
        row = citra.shape[0]
        col = citra.shape[1]
        # Ekstraksi Kanal yang diperlukan
        citraH, citraS, citraV = ekstrakPisahCitra(citra, "hsv")
        citraR, citraG, citraB = ekstrakPisahCitra(citra, "rgb")
        citraGray = ubahCitraGray(citra)
        ambangPlasmaHasil, ambangIntiHasil = ambangCitra(
            citraH, citraS, citraV)
        citraKonturPlasma, citraKonturInti, citraHullPlasma, citraHullInti, citraAreaHullPlasma, citraAreaHullInti, konturInti = konturHullCitra(
            ambangPlasmaHasil, ambangIntiHasil)
        citraModPlasmaFix, citraModIntiFix, citraModPlasma = detailCitra(
            ambangPlasmaHasil, ambangIntiHasil, citraV)
        citraModPlasmaR, citraModPlasmaG, citraModPlasmaB = ambangWarnaCitra(
            citraModPlasma, citraR, citraG, citraB)
        citraModIntiR, citraModIntiG, citraModIntiB = ambangWarnaCitra(
            ambangIntiHasil, citraR, citraG, citraB)
        citraPlasmaGray = ekstrakAbu(ambangPlasmaHasil, citraGray)
        citraIntiGray = ekstrakAbu(ambangIntiHasil, citraGray)

        # Perhitungan fitur-fitur
        luasPlasma = ekstrakPikselPutih(ambangPlasmaHasil)
        luasInti = ekstrakPikselPutih(ambangIntiHasil)
        kelilingPlasma = ekstrakPikselPutih(citraKonturPlasma)
        kelilingInti = ekstrakPikselPutih(citraKonturInti)
        luasConvexPlasma = ekstrakPikselPutih(citraAreaHullPlasma)
        luasConvexInti = ekstrakPikselPutih(citraAreaHullInti)
        solidityPlasma = bagiVariabel(luasPlasma, luasConvexPlasma)
        solidityInti = bagiVariabel(luasInti, luasConvexInti)
        rerataPlasma, rerataPlasmaR, rerataPlasmaG, rerataPlasmaB, sdPlasma = rerataSD(
            citraModPlasmaFix, citraModPlasmaR, citraModPlasmaG, citraModPlasmaB)
        rerataInti, rerataIntiR, rerataIntiG, rerataIntiB, sdInti = rerataSD(
            citraModIntiFix, citraModIntiR, citraModIntiG, citraModIntiB)
        circularityPlasma = circularity(luasPlasma, kelilingPlasma)
        circularityInti = circularity(luasInti, kelilingInti)
        liLp = bagiVariabel(luasInti, luasPlasma)
        kiKp = bagiVariabel(kelilingInti, kelilingPlasma)
        luasNormalInti, kelilingNormalInti = normalisasiInti(
            luasInti, kelilingInti, col, row)
        eccentricity = eccentricityCitra(konturInti)
        entropiInti, energiInti, kontrasInti, homogenitasInti = teksturCitra(
            citraIntiGray)
        entropiPlasma, energiPlasma, kontrasPlasma, homogenitasPlasma = teksturCitra(
            citraPlasmaGray)

        # Simpan ke dalam list
        fitur = [luasInti, kelilingInti, solidityInti, sdInti, circularityInti, rerataIntiR, rerataIntiG, rerataIntiB, entropiInti, energiInti, kontrasInti, homogenitasInti, luasPlasma, kelilingPlasma, solidityPlasma,
                 sdPlasma, circularityPlasma, rerataPlasmaR, rerataPlasmaG, rerataPlasmaB, entropiPlasma, energiPlasma, kontrasPlasma, homogenitasPlasma, luasNormalInti, kelilingNormalInti, eccentricity, liLp, kiKp]

        # Cetak Hasil Ekstraksi Fitur
        print("Legit lah")
        print(fitur)

        return fitur

    def __del__(self):
        print('Proses ekstraksi fitur selesai.')

def ekstrakPisahCitra(citra, mode):
    a = citra
    b = citra
    c = citra
    if mode == "hsv":
        citra = cv2.cvtColor(citra, cv2.COLOR_BGR2HSV)
        a, b, c = cv2.split(citra)
    # Jika memilih selain HSV, asumsi RGB
    else:
        a, b, c = cv2.split(citra)
    return a, b, c


def ubahCitraGray(citra):
    return cv2.cvtColor(citra, cv2.COLOR_BGR2GRAY)


def ambangCitra(citraH, citraS, citraV):
    tPlasmaH = 130
    tPlasmaS = 27
    tPlasmaV = 62
    tIntiS = 149
    ret1, ambangPlasmaH = cv2.threshold(citraH, tPlasmaH, 255, cv2.THRESH_BINARY)
    ret2, ambangPlasmaS = cv2.threshold(citraS, tPlasmaS, 255, cv2.THRESH_BINARY)
    ret3, ambangPlasmaV = cv2.threshold(citraV, tPlasmaV, 255, cv2.THRESH_BINARY)
    ret4, ambangIntiHasil = cv2.threshold(citraS, tIntiS, 255, cv2.THRESH_BINARY)
    ambangPlasmaHasil = cv2.bitwise_and(ambangPlasmaH, ambangPlasmaS)
    ambangPlasmaHasil = cv2.bitwise_and(ambangPlasmaV, ambangPlasmaHasil)
    return ambangPlasmaHasil, ambangIntiHasil


def ekstrakPikselPutih(citra):
    return cv2.countNonZero(citra)


def bagiVariabel(i, p):
    return float(i / p)


def konturHullCitra(plasma, inti):
    imgP, konturPlasma, hierarkiKonturPlasma = cv2.findContours(
        plasma, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    imgI, konturInti, hierarkiKonturInti = cv2.findContours(
        inti, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    row = plasma.shape[0]
    col = plasma.shape[1]

    citraKonturPlasma = np.zeros((row, col), dtype=np.uint8)
    citraKonturInti = np.zeros((row, col), dtype=np.uint8)
    citraHullPlasma = np.zeros((row, col), dtype=np.uint8)
    citraHullInti = np.zeros((row, col), dtype=np.uint8)
    citraAreaHullPlasma = np.zeros((row, col), dtype=np.uint8)
    citraAreaHullInti = np.zeros((row, col), dtype=np.uint8)
    hullPlasma = []
    hullInti = []

    for i in range(len(konturPlasma)):
        hullPlasma.append(cv2.convexHull(konturPlasma[i]))

    for i in range(len(konturInti)):
        hullInti.append(cv2.convexHull(konturInti[i]))
        # hullInti[i] = cv2.convexHull(konturInti[i])

    for i in range(len(konturPlasma)):
        cv2.drawContours(citraKonturPlasma, konturPlasma,
                         i, (255, 255, 255), 1, 8)
        cv2.drawContours(citraHullPlasma, hullPlasma, i, (255, 255, 255), 1, 8)
        # titikHullPlasma = []
        # j = 0
        # abal = len(hullPlasma[0])
        # for j in range(len(hullPlasma[i])):
        #     titikHullPlasma.append(hullPlasma[i,j])
        #     # titikHullPlasma[j] = hullPlasma[i,j]
        # jumlahHullPlasma = np.array(titikHullPlasma, np.int32)
        # cv2.fillPoly(citraAreaHullPlasma, jumlahHullPlasma, (255, 255, 255), 8)
    cv2.fillPoly(citraAreaHullPlasma, konturPlasma, (255, 255, 255), 8)

    for i in range(len(konturInti)):
        citraKonturInti = cv2.drawContours(
            citraKonturInti, konturInti, i, (255, 255, 255), 1, 8)
        citraHullInti = cv2.drawContours(
            citraHullInti, hullInti, i, (255, 255, 255), 1, 8)
        # titikHullInti[200] = None
        # j = None
        # for j in range(len(hullInti[i])):
        #     titikHullInti[j] = hullInti[i, j]
        # jumlahHullInti = np.array(titikHullInti, np.int32)
        # cv2.fillPoly(citraAreaHullInti, jumlahHullInti, (255, 255, 255), 8)
    cv2.fillPoly(citraAreaHullInti, konturInti, (255, 255, 255), 8)
    return citraKonturPlasma, citraKonturInti, citraHullPlasma, citraHullInti, citraAreaHullPlasma, citraAreaHullInti, konturInti


def detailCitra(plasma, inti, v):
    citraModPlasma = cv2.bitwise_not(inti)
    citraModPlasma = cv2.bitwise_and(citraModPlasma, plasma)
    citraModPlasmaFix = cv2.bitwise_and(citraModPlasma, v)
    citraModIntiFix = cv2.bitwise_and(inti, v)
    return citraModPlasmaFix, citraModIntiFix, citraModPlasma


def ambangWarnaCitra(citra, r, g, b):
    hasilR = cv2.bitwise_and(citra, r)
    hasilG = cv2.bitwise_and(citra, g)
    hasilB = cv2.bitwise_and(citra, b)
    return hasilR, hasilG, hasilB


def ekstrakAbu(citra, gray):
    return cv2.bitwise_and(citra, gray)


def rerataSD(citra, r, g, b): #TODO buat jadi integer, bukan array
    rer, sd = cv2.meanStdDev(citra)
    rerR, sdR = cv2.meanStdDev(r)
    rerG, sdG = cv2.meanStdDev(g)
    rerB, sdB = cv2.meanStdDev(b)
    rerata = float(rer[0])
    rerataR = float(rerR[0])
    rerataG = float(rerG[0])
    rerataB = float(rerB[0])
    sD = float(sd[0])
    return rerata, rerataR, rerataG, rerataB, sD


def circularity(l, k):
    return 4 * math.pi * l / pow(k, 2)


def normalisasiInti(l, k, c, r):
    luasNormal = float(l / (c * r))
    kelilingNormal = float(k / (2 * (c + r)))
    return luasNormal, kelilingNormal


def eccentricityCitra(kontur):
    luasKontur = 0
    terluasKontur = 0
    indeks = 0
    myu20 = 0.00
    myu11 = 0.00
    myu02 = 0.00
    eccentricityCitra = 0.00

    for i in range(len(kontur)):
        luasKontur = cv2.contourArea(kontur[i], False)
        if luasKontur > terluasKontur:
            terluasKontur = luasKontur
            indeks = i
    mu = cv2.moments(kontur[indeks], False)
    myu20 = myu20 + mu['mu20']
    myu02 = myu02 + mu['mu02']
    myu11 = myu11 + mu['mu11']

    matriks = np.array([[myu20, myu11], [myu11, myu02]], dtype=np.float32) 
    ret, eigenv, eigenvct = cv2.eigen(matriks) # TODO error: (-215) type == CV_32F || type == CV_64F
    eigenv1 = eigenv[0, 0]
    eigenv2 = eigenv[1, 0]

    # Hitung eigen value
    if eigenv1 >= eigenv2:
        eccentricityCitra = eigenv2 / eigenv1
    else:
        eccentricityCitra = eigenv1 / eigenv2
    return eccentricityCitra


def teksturCitra(citra):
    row = citra.shape[0]
    col = citra.shape[1]
    entropi = 0
    energi = 0
    kontras = 0
    homogenitas = 0
    gl = np.zeros((256, 256), np.uint8)
    a = 0
    b = 0
    for i in range(0, row):
        for j in range(0, col - 1):
            a = citra[i, j]
            b = citra[i, j + 1]
            gl[a, b] = gl[a, b] + 1

    gl = gl + gl.T
    gl = gl / cv2.sumElems(gl)[0]

    for i in range(0, 256):
        for j in range(0, 256):
            energi = energi + (gl[i, j] * gl[i, j])
            kontras = kontras + (i - j) * (i - j) * gl[i, j]
            homogenitas = homogenitas + gl[i, j] / (1 + abs(i - j))
            if (gl[i, j] != 0):
                entropi = entropi - gl[i, j] * math.log10(gl[i, j])

    return entropi, energi, kontras, homogenitas
