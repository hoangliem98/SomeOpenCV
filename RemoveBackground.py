import numpy as np
import cv2 as cv

import sys

class App():
    BLUE = [255,0,0]        
    BLACK = [0,0,0]         
    WHITE = [255,255,255]   

    DRAW_BG = {'color' : BLACK, 'val' : 0}
    DRAW_FG = {'color' : WHITE, 'val' : 1}

    rect = (0,0,1,1)
    drawing = False         
    rectangle = False       
    rect_over = False       
    rect_or_mask = 100      
    value = DRAW_FG         
    thickness = 3           

    def onmouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.rectangle = True
            self.ix, self.iy = x,y

        elif event == cv.EVENT_MOUSEMOVE:
            if self.rectangle == True:
                self.img = self.img2.copy()
                cv.rectangle(self.img, (self.ix, self.iy), (x, y), self.BLUE, 2)
                self.rect = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
                self.rect_or_mask = 0

        elif event == cv.EVENT_LBUTTONUP:
            self.rectangle = False
            self.rect_over = True
            cv.rectangle(self.img, (self.ix, self.iy), (x, y), self.BLUE, 2)
            self.rect = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
            self.rect_or_mask = 0
            print(" Nhấn 'c' và đợi 1 lát \n")

    def onmouse_fix(self, event, x, y, flags, param):
        if event == cv.EVENT_RBUTTONDOWN:
            if self.rect_over == False:
                print("Ảnh chưa được tách nền vui lòng khoanh vùng đối tượng trước \n")
            else:
                self.drawing = True
                cv.circle(self.img, (x,y), self.thickness, self.value['color'], -1)
                cv.circle(self.mask, (x,y), self.thickness, self.value['val'], -1)

        elif event == cv.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv.circle(self.img, (x, y), self.thickness, self.value['color'], -1)
                cv.circle(self.mask, (x, y), self.thickness, self.value['val'], -1)

        elif event == cv.EVENT_RBUTTONUP:
            if self.drawing == True:
                self.drawing = False
                cv.circle(self.img, (x, y), self.thickness, self.value['color'], -1)
                cv.circle(self.mask, (x, y), self.thickness, self.value['val'], -1)

    def run(self):
        image = cv.imread('1.jpg')
        self.img = cv.resize(src=image, dsize=(300, 400))
        self.img2 = self.img.copy()                          
        self.mask = np.zeros(self.img.shape[:2], np.uint8)
        self.output = np.zeros(self.img.shape, np.uint8)           

        # input and output windows
        cv.namedWindow('output')
        cv.namedWindow('input')
        cv.setMouseCallback('input', self.onmouse)
        
        cv.setMouseCallback('output', self.onmouse_fix)

        print(" Giữ kéo chuột trái để khoanh vùng bố cục chính \n")

        while True:

            cv.imshow('output', self.output)
            cv.imshow('input', self.img)
            k = cv.waitKey(1)

            if k == ord('q'): # press 'q' to exit
                break
            elif k == ord('0'): # BG drawing
                print(" Giữ và rê chuột phải trên output để xóa những vùng background thừa \n")
                self.value = self.DRAW_BG
            elif k == ord('1'): # FG drawing
                print(" Giữ và rê chuột phải trên output để làm rõ một số chi tiết bị xóa theo nền \n")
                self.value = self.DRAW_FG
            elif k == ord('s'):
                cv.imwrite('test_output.png', self.output)
                print(" Lưu thành công \n")
            elif k == ord('r'): # reset everything
                print("Reset ảnh \n")
                self.rect = (0,0,1,1)
                self.drawing = False
                self.rectangle = False
                self.rect_or_mask = 100
                self.rect_over = False
                self.value = self.DRAW_FG
                self.img = self.img2.copy()
                self.mask = np.zeros(self.img.shape[:2], dtype = np.uint8) 
                self.output = np.zeros(self.img.shape, np.uint8)           
            elif k == ord('c'): 
                print(""" Để chỉnh sửa chi tiết ảnh, chọn các nút 0 hoặc 1 rồi nhấn lại 'c' \n""")
                try:
                    bgdmodel = np.zeros((1, 65), np.float64)
                    fgdmodel = np.zeros((1, 65), np.float64)
                    if (self.rect_or_mask == 0):        
                        cv.grabCut(self.img2, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_RECT)
                        self.rect_or_mask = 1
                    elif (self.rect_or_mask == 1):      
                        cv.grabCut(self.img2, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_MASK)
                except:
                    import traceback
                    traceback.print_exc()

            mask2 = np.where((self.mask==1) + (self.mask==3), 255, 0).astype('uint8')
            self.output = cv.bitwise_and(self.img2, self.img2, mask=mask2)

if __name__ == '__main__':
    print(__doc__)
    App().run()
    cv.destroyAllWindows()
