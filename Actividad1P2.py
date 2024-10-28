import cv2 as cv
import numpy as np
import math

img = cv.imread('bob.png', 0)


x, y = img.shape

result_img = np.zeros((int(x * 2), int(y * 2)), dtype=np.uint8)

angle1 = -30
theta1 = math.radians(angle1)

angle2 = 60
theta2 = math.radians(angle2)

scale = 2

for i in range(int(x * scale)):
    for j in range(int(y * scale)):

        orig_x = int(i / scale)
        orig_y = int(j / scale)


        rotated_x1 = int((orig_x - x // 2) * math.cos(theta1) - (orig_y - y // 2) * math.sin(theta1) + x // 2)
        rotated_y1 = int((orig_x - x // 2) * math.sin(theta1) + (orig_y - y // 2) * math.cos(theta1) + y // 2)

        rotated_x2 = int((rotated_x1 - x // 2) * math.cos(theta2) - (rotated_y1 - y // 2) * math.sin(theta2) + x // 2)
        rotated_y2 = int((rotated_x1 - x // 2) * math.sin(theta2) + (rotated_y1 - y // 2) * math.cos(theta2) + y // 2)

        if 0 <= rotated_x2 < x and 0 <= rotated_y2 < y:
            result_img[i, j] = img[rotated_x2, rotated_y2]


cv.imshow('Original Image', img)
cv.imshow('Rotated, Rotated, and Scaled Image', result_img)
cv.waitKey(0)
cv.destroyAllWindows()