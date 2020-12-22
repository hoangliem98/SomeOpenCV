import cv2 as cv
import pytesseract
from scipy import ndimage

file = "savetest.jpg"
  
pytesseract.pytesseract.tesseract_cmd = 'D:\\LN\\Tesseract\\tesseract.exe'
  
def getTextNumber(img):
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

   
img = cv.imread(file) 
imgTmp = ndimage.rotate(img, 180)
imgs = [img, imgTmp]
maxText = 0
for i in imgs:
    if(getTextNumber(i) > maxText):
        maxText = getTextNumber(i)
        img = i

cv.imshow('max', img)
cv.waitKey(0)





#def getTextNumber(img):  
#    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  
#    ret, thresh1 = cv.threshold(gray, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)  
#    rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (18, 18)) 
#    print(thresh1)
#    ilation = cv.dilate(thresh1, rect_kernel, iterations = 1)  
#    contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 
#    img2 = img.copy() 
#    for cnt in contours: 
#        x, y, w, h = cv.boundingRect(cnt)   
#        rect = cv.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)    
#        cropped = img2[y:y + h, x:x + w]    
#        text = pytesseract.image_to_string(cropped) 
#    return len(text)

#img = cv.imread(file) 
#imgtmp = ndimage.rotate(img, 180)
#imgs = [img, imgtmp]
##for i in imgs:
#print(getTextNumber(img))

