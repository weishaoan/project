import cv2
import numpy as np
kernel = np.ones((10,10),np.uint8)
kernel2 = np.ones((12,12),np.uint8)
img = cv2.imread('project/img/WTF.jpg')
blur = cv2.GaussianBlur(img,(15,15),10)
canny = cv2.Canny(img,150,200)
dilate = cv2.dilate(canny,kernel,iterations=1)
erode = cv2.erode(dilate,kernel2,iterations=2)


cv2.imshow('Image',img)
cv2.imshow('Blur',blur)
cv2.imshow('Canny',canny)
cv2.imshow('Dilate',dilate)
cv2.imshow('Erode',erode)
cv2.waitKey(0)