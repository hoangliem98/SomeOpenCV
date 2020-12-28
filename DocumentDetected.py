import cv2 as cv
import numpy as np
import math
import pytesseract
from scipy import ndimage
import matplotlib.pyplot as plt
import statistics
import argparse
#import imutils

brown = 19,69,139

#lấy số lượng chữ lấy ra từ ảnh
def getTextNumber(img):
    pytesseract.pytesseract.tesseract_cmd = 'D:\\LN\\Tesseract\\tesseract.exe'
    ret, thresh1 = cv.threshold(img, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)   
    rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (18, 18))   
    dilation = cv.dilate(thresh1, rect_kernel, iterations = 1)  
    contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 
  
    im2 = img.copy() 
    sumtest = 0
  
    for cnt in contours: 
        print("Đang xử lý...")
        x, y, w, h = cv.boundingRect(cnt)    
        rect = cv.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)    
        cropped = im2[y:y + h, x:x + w]    
        text = pytesseract.image_to_string(cropped, lang="eng")
        for t in range(len(text)):
            sumtest += t
    return sumtest

#Xoay ảnh thẳng đứng
def rotate(img): 
    imgThres = preProcessing(img)    
    biggest = getContours(imgThres)
    newPoints = reorder(biggest)
    if ((newPoints[1][0][0] - newPoints[0][0][0]) < (newPoints[2][0][1] - newPoints[0][0][1])):
        line1 = np.concatenate((newPoints[0], newPoints[1]),axis=1)
        line2 = np.concatenate((newPoints[2], newPoints[3]),axis=1)
    else:       
        line1 = np.concatenate((newPoints[0], newPoints[2]),axis=1)
        line2 = np.concatenate((newPoints[1], newPoints[3]),axis=1)
    lines = np.concatenate((line1, line2))
    angles = []

    for [x1, y1, x2, y2] in lines:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    median_angle = np.median(angles)
    print("angle", median_angle)
    img = ndimage.rotate(img, median_angle)
    return img

#Kiểm tra xem số lượng chữ ở chiều nào lọc được chính xác hơn thì lấy chiều đó làm chính
#Hàm này làm chậm chương trình do scan ảnh thành text sau đó đếm text
def checkTextRotate(img): 
    imgTmp = ndimage.rotate(img, 180)
    imgs = [img, imgTmp]
    maxText = 0
    image = img.copy()
    for i in imgs:
        if(getTextNumber(i) >= maxText):
            maxText = getTextNumber(i)
            image = i
            print("Maxtext:",maxText)   
    print("Đã xong")  
    return image 

#def preProcessing(img):
#    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#    imgBlur = cv.GaussianBlur(img,(5,5),1)
#    imgCanny = cv.Canny(imgBlur,200,200)
#    kernel = np.ones((5,5))
#    imgDial = cv.dilate(imgCanny,kernel,iterations=2)
#    imgThres = cv.erode(imgDial,kernel,iterations=1)
    
#    return imgThres

def preProcessing(img):  
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    invGamma = 1.0 / 0.3
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    gray = cv.LUT(gray, table)
    ret,thresh1 = cv.threshold(gray,80,255,cv.THRESH_BINARY)
    return thresh1

def getContours(img):
    biggest = np.array([])
    max_area = 0 
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
            area = cv.contourArea(cnt)
            if area > 100:
                #cv.drawContours(imgContour, cnt, -1, (255,255,0), 3)
                peri = cv.arcLength(cnt,True)
                approx = cv.approxPolyDP(cnt,0.1*peri,True)
                if area > max_area and len(approx)==4:
                        biggest = approx
                        max_area = area
    cv.drawContours(imgContour, biggest, -1, (0,255,0),20)
    return biggest

#def getContours(img):
#    biggest = np.array([])
#    maxArea = 0
#    contours,hierarchy = cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
#    for cnt in contours:
#        area = cv.contourArea(cnt)
#        if area > 5000:
#            #cv.drawContours(imgContour, cnt, -1, (255,255,0), 3)
#            peri = cv.arcLength(cnt,True)
#            approx = cv.approxPolyDP(cnt,0.02*peri,True)
#            print(len(approx))
#            if area > maxArea and len(approx) == 4:
#                biggest = approx
#                maxArea = area

#    cv.drawContours(imgContour, biggest, -1, (255,0,0), 20)
#    return biggest

def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    #print(myPoints[1][1] - myPoints[0][1])
    #print(myPoints)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=-1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def unsharp_mask(img, amount=1.0, threshold=0):
    blurred = cv.medianBlur(img, 5)
    sharpened = float(amount + 1) * img - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(img - blurred) < threshold
        np.copyto(sharpened, img, where=low_contrast_mask)
    return sharpened

def getWarp(img, biggest):
    biggest = reorder(biggest)
    top = biggest[1][0][0] - biggest[0][0][0]
    bottom = biggest[3][0][0] - biggest[2][0][0]
    left = biggest[2][0][1] - biggest[0][0][1]
    right = biggest[3][0][1] - biggest[1][0][1]
    width = round(statistics.mean([top, bottom]))
    height = round(statistics.mean([left, right]))
    pst1 = np.float32(biggest)
    pst2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv.getPerspectiveTransform(pst1,pst2)
    imgOutput = cv.warpPerspective(img, matrix, (width,height))
    #upSharp = unsharp_mask(imgOutput)
    brightness = scanBrightness(imgOutput)
    return brightness

def scanBrightness(img):
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dilated_img = cv.dilate(imgGray, np.ones((7,7), np.uint8)) 
    #dilated_img = unsharp_mask(imgGray)
    bg_img = cv.medianBlur(dilated_img,21)
    diff_img = 255 - cv.absdiff(imgGray, bg_img)
    norm_img = diff_img.copy() 
    cv.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
    _, thr_img = cv.threshold(norm_img, 230, 0, cv.THRESH_TRUNC)
    imgGray = cv.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
    return imgGray

#main
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="Đường dẫn đến ảnh muốn nhận dạng")
args = vars(ap.parse_args())
#fileImg = imutils.rotate(cv.imread(file),30)
img = cv.imread(args["image"])
#img = cv.copyMakeBorder(fileImg,20,100,20,270,cv.BORDER_CONSTANT,value=brown)

imgContour = img.copy()
img = rotate(img)
imgThres = preProcessing(img)
biggest = getContours(imgThres)
if len(biggest) == 0:
    print("Bad Image")
else:
    imgWarped = getWarp(img, biggest)
    rotateWarper = checkTextRotate(imgWarped)
    unsharpFinal = unsharp_mask(rotateWarper)

    #cv.imshow("result", rotateWarper)
    cv.imwrite(args["image"], rotateWarper)
    #cv.imshow("rotate", img)
    cv.waitKey(0)