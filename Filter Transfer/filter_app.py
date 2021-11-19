import cv2
from skimage import exposure

reference = cv2.imread("/Users/shireen/Documents/opencv/Learning/building2.jpeg")
source = cv2.imread("/Users/shireen/Documents/opencv/Learning/building1.jpeg")

multi = True if source.shape[-1] > 1 else False
matched = exposure.match_histograms(source,reference,multichannel=multi)

cv2.imshow("Matched",matched)
cv2.waitKey(0)


