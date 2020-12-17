import cv2 as cv
import numpy as np
#import imutils

width, height = 540, 720 
brown = 19,69,139

def preProcessing(img):
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(img,(5,5),1)
    imgCanny = cv.Canny(imgBlur,200,200)
    kernel = np.ones((5,5))
    imgDial = cv.dilate(imgCanny,kernel,iterations=2)
    imgThres = cv.erode(imgDial,kernel,iterations=1)
    
    return imgThres

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours,hierarchy = cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 5000:
            #cv.drawContours(imgContour, cnt, -1, (255,255,0), 3)
            peri = cv.arcLength(cnt,True)
            approx = cv.approxPolyDP(cnt,0.02*peri,True)
            print(len(approx))
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area

    cv.drawContours(imgContour, biggest, -1, (255,0,0), 20)
    return biggest

def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    #if (myPoints[1][1] - myPoints[0][1]) > (myPoints[2][0] - myPoints[1][0]):
    #    print(myPoints[1] - myPoints[0])
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    print(myPoints[1][1] - myPoints[0][1])

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=-1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def getWarp(img, biggest):
    biggest = reorder(biggest)
    pst1 = np.float32(biggest)
    pst2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv.getPerspectiveTransform(pst1,pst2)
    imgOutput = cv.warpPerspective(img, matrix, (width,height))
    return imgOutput

file = "doc2.jpg"
#ileImg = imutils.rotate(cv.imread(file),30)
img = cv.imread(file)
#img = cv.copyMakeBorder(fileImg,20,100,20,270,cv.BORDER_CONSTANT,value=brown)

imgContour = img.copy()
imgThres = preProcessing(img)
biggest = getContours(imgThres)
imgWarped = getWarp(img, biggest)

cv.imshow(file, imgWarped)
cv.imwrite("savetest.jpg", imgWarped)
cv.imshow("originale", imgContour)
cv.waitKey(0)




#img = np.zeros((512,550,3), np.uint8)
#img[:] = 255,255,255

#print(img.shape[1])
##cv.line(img,(0,0),(img.shape[1],img.shape[0]),(255,255,0),3)
#cv.rectangle(img,(10,50),(20,10),(225,225,0),3)

#cv.imshow("Image", img)
#cv.waitKey(0)
