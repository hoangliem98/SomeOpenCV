import cv2 as cv
import pytesseract
from PIL import Image

file = "savetest.jpg"
im = Image.open(file)
im.rotate(90).save(file)
  
pytesseract.pytesseract.tesseract_cmd = 'D:\\LN\\Tesseract\\tesseract.exe'


img = cv.imread(file) 
  
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
      
    print(text)
