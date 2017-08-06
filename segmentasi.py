import numpy as np
import cv2

wbc_cascade = cv2.CascadeClassifier('cascade_wbc_10.xml')
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
img = cv2.imread('25_w_154.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

wbc = wbc_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(225,225))
for (x,y,w,h) in wbc:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    # roi_gray = gray[y:y+h, x:x+w]
    # roi_color = img[y:y+h, x:x+w]
    # eyes = eye_cascade.detectMultiScale(roi_gray)
    # for (ex,ey,ew,eh) in eyes:
    #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
hasil = cv2.resize(img, (640,480))
cv2.imshow("hasil", hasil)
# cv2.imshow('img',img, cv2.WINDOW_NORMAL)
cv2.waitKey(0)
cv2.destroyAllWindows()