import cv2 as cv
import numpy as np 

img = np.ones((500,500, 3), dtype = np.uint8)*255 #lienzo en balnco

for i in range(400):
    cv.circle(img, (i, i),50, (0,234,21), -1)
    cv.imshow ('img ', img)
    img = np.ones((500, 500, 3 ), dtype=np.uint8)*255
    cv.waitKey(40)

cv.destroyAllWindows
    
