import cv2 as cv
import numpy as np
import math
import pytesseract
from scipy import ndimage
#import imutils

width, height = 640, 780 
brown = 19,69,139

#lấy số lượng chữ lấy ra từ ảnh
def getTextNumber(img):
    pytesseract.pytesseract.tesseract_cmd = 'D:\\LN\\Tesseract\\tesseract.exe'
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  
    ret, thresh1 = cv.threshold(gray, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)   
    rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (18, 18))   
    dilation = cv.dilate(thresh1, rect_kernel, iterations = 1)  
    contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 
  
    im2 = img.copy() 
  
    for cnt in contours: 
        x, y, w, h = cv.boundingRect(cnt)    
        rect = cv.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)    
        cropped = im2[y:y + h, x:x + w]    
        text = pytesseract.image_to_string(cropped) 
    return len(text)

#Xoay ảnh thẳng đứng
def rotate(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_edges = cv.Canny(img_gray, 100, 100, apertureSize=3)
    lines = cv.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

    angles = []

    for [[x1, y1, x2, y2]] in lines:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    median_angle = np.median(angles)
    print("angle", median_angle)
    img = ndimage.rotate(img, median_angle)
    return img

#Kiểm tra xem số lượng chữ ở chiều nào lọc được chính xác hơn thì lấy chiều đó làm chính
def checkTextRotate(img): 
    imgTmp = ndimage.rotate(img, 180)
    imgs = [img, imgTmp]
    maxText = 0
    for i in imgs:
        if(getTextNumber(i) > maxText):
            maxText = getTextNumber(i)
            img = i
    return img

#Lọc ảnh để lấy phần document
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
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    print(myPoints[1][1] - myPoints[0][1])

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=-1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def getWarp(img, biggest):
    biggest = reorder(biggest)
    pst1 = np.float32(biggest)
    pst2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv.getPerspectiveTransform(pst1,pst2)
    imgOutput = cv.warpPerspective(img, matrix, (width,height))
    outputSharpen = unsharp_mask(imgOutput)
    return outputSharpen

file = "doc2.jpg"
#fileImg = imutils.rotate(cv.imread(file),30)
img = cv.imread(file)
img = rotate(img)
#img = cv.copyMakeBorder(fileImg,20,100,20,270,cv.BORDER_CONSTANT,value=brown)

imgContour = img.copy()
imgThres = preProcessing(img)
biggest = getContours(imgThres)
if len(biggest) == 0:
    print("Bad Image")
else:
    imgWarped = getWarp(img, biggest)
    imgWarped = checkTextRotate(imgWarped)

    cv.imshow(file, imgWarped)
    cv.imwrite("savetest.jpg", imgWarped)
    cv.imshow("original", imgContour)
    cv.waitKey(0)