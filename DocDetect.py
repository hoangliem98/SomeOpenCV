import cv2
import numpy as np

def preProcessing(img):  
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    invGamma = 1.0 / 0.3
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    gray = cv2.LUT(gray, table)
    ret,thresh1 = cv2.threshold(gray,80,255,cv2.THRESH_BINARY)
    return thresh1

def getContour(img):
    biggest = np.array([])
    max_area = 0 
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                #cv.drawContours(imgContour, cnt, -1, (255,255,0), 3)
                peri = cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,0.1*peri,True)
                if area > max_area and len(approx)==4:
                        biggest = approx
                        max_area = area
    cv2.drawContours(imgContour, biggest, -1, (0,255,0),20)
    return biggest


img = cv2.imread('doc2.jpg')
imgContour = img.copy()
imgThres = preProcessing(img)
biggest = getContour(imgThres)
cv2.imshow('hola.png', imgContour)
cv2.waitKey(0)
