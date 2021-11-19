import cv2
import numpy as np



image = cv2.imread('/Users/shireen/Documents/opencv/Learning/coins.jpeg')
cv2.imshow("Original",image)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray,(5,5),0)

edged = cv2.Canny(blurred,10,250)
cv2.imshow("Final",edged)

(cnts,_) = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print(f"There are {len(cnts)} in the image")

cv2.waitKey(0)
