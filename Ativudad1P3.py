import cv2 as cv
import numpy as np
import math

img = cv.imread('bob.png', 0)


x, y = img.shape


result_img = np.zeros((int(x * 2), int(y * 2)), dtype=np.uint8)


angle = 70
theta = math.radians(angle)


dx, dy = 20, 20


scale = 2


for i in range(int(x * scale)):
    for j in range(int(y * scale)):

        orig_x = int(i / scale)
        orig_y = int(j / scale)

        rotated_x = int((orig_x - x // 2) * math.cos(theta) - (orig_y - y // 2) * math.sin(theta) + x // 2)
        rotated_y = int((orig_x - x // 2) * math.sin(theta) + (orig_y - y // 2) * math.cos(theta) + y // 2)

        translated_x = rotated_x + dx
        translated_y = rotated_y + dy

        if 0 <= translated_x < x and 0 <= translated_y < y:
            result_img[i, j] = img[translated_x, translated_y]

cv.imshow('Original Image', img)
cv.imshow('Rotated, Translated, and Scaled Image', result_img)
cv.waitKey(0)
cv.destroyAllWindows()