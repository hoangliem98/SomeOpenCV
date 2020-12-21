import cv2
import numpy as np


cv2.startWindowThread()
cv2.namedWindow("Original")
cv2.namedWindow("Sharpen")

imgIn = cv2.imread("savetest.jpg")
cv2.imshow("Original", imgIn)



kernel = np.zeros( (9,9), np.float32)
kernel[4,4] = 2.0   

boxFilter = np.ones( (9,9), np.float32) / 81.0

kernel = kernel - boxFilter

custom = cv2.filter2D(imgIn, -1, kernel)
cv2.imshow("Sharpen", custom)


cv2.waitKey(0)
