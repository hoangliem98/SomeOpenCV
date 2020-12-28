import cv2
import numpy as np


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image,(5,5),0)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

img = cv2.imread('savetest.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dilated_img = unsharp_mask(img)
bg_img = cv2.medianBlur(dilated_img,21)
diff_img = 255 - cv2.absdiff(img, bg_img)
norm_img = diff_img.copy() 
cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
_, thr_img = cv2.threshold(norm_img, 230, 0, cv2.THRESH_TRUNC)
img = cv2.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
cv2.imshow("s", bg_img)
cv2.imshow("ss", img)
cv2.waitKey(0)