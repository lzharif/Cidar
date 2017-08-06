# Thresholding Test

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('gradient.png',0)

def ambangCitraGlobal(self, citraH, citraS, citraV):
    tPlasmaH = 130 
    tPlasmaS = 27
    tPlasmaV = 62
    tIntiS = 149
    ambangPlasmaH = cv2.threshold(citraH, tPlasmaH, 255, cv2.THRESH_BINARY)
    ambangPlasmaS = cv2.threshold(citraS, tPlasmaS, 255, cv2.THRESH_BINARY)
    ambangPlasmaV = cv2.threshold(citraV, tPlasmaV, 255, cv2.THRESH_BINARY)
    ambangIntiHasil = cv2.threshold(citraS, tIntiS, 255, cv2.THRESH_BINARY)
    ambangPlasmaHasil = cv2.bitwise_and(ambangPlasmaH, ambangPlasmaS)
    ambangPlasmaHasil = cv2.bitwise_and(ambangPlasmaV, ambangPlasmaHasil)
    return ambangPlasmaHasil, ambangIntiHasil

def ambangCitraAdaptif(self, citraH, citraS, citraV):
    ambangPlasmaH = cv2.adaptiveThreshold(citraH, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    ambangPlasmaS = cv2.threshold(citraS, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    ambangPlasmaV = cv2.threshold(citraV, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    ambangIntiHasil = cv2.threshold(citraS, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    ambangPlasmaHasil = cv2.bitwise_and(ambangPlasmaH, ambangPlasmaS)
    ambangPlasmaHasil = cv2.bitwise_and(ambangPlasmaV, ambangPlasmaHasil)
    return ambangPlasmaHasil, ambangIntiHasil