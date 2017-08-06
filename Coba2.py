# Coba-Coba

import cv2
import sys
import numpy as np

if len(sys.argv)!=2:                  ## Check for error in usage syntax
    print("Usage : python display_image.py <image_file>")

else:
    img = cv2.imread(sys.argv[1],cv2.IMREAD_COLOR)  ## Read image file
    cv2.namedWindow('Display Window')        ## create window for display
    cv2.imshow('Display Window',img)         ## Show image in the window
    print("size of image: ",img.shape)        ## print size of image
    cv2.waitKey(0)                           ## Wait for keystroke
    cv2.destroyAllWindows()                  ## Destroy all windows

