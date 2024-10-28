import cv2 
import numpy as np

width, height = 640, 480


img = np.zeros((height, width, 3), np.uint8)

radius = 20
x, y = width//2, height//2 
speed_x, speed_y = 5, 5

def draw_ball(img, x,y, radius ):
    cv2.circle(img, (x,y), radius, (0, 255, 0), -1)
    
    while True:
     img[:]= 0
     
     
     x += speed_x
     y += speed_y
     
     if x + radius >= width or x - radius <= 0:
         speed_x *= -1
     if y + radius >= height or y - radius <= 0:
             speed_y *= -1
    
     draw_ball(img, x, y, radius ) 
     
     cv2.imshow('Pelota rebotando', img) 
     if cv2.waitKey(1) == ord('q'):
         break           

cv2.destroyAllWindows()