# -*- coding: UTF-8 -*-
__author__ = 'Administrator'

import cv2
import numpy as np

img = cv2.imread('1.jpg', 0)
img1 = cv2.imread('1.jpg')
kernel = np.ones((2, 2), np.int8)
kernel1 = np.ones((3, 3), np.int8)
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
close = cv2.morphologyEx(th2,cv2.MORPH_CLOSE, kernel1)
open = cv2.morphologyEx(th2,cv2.MORPH_OPEN, kernel)
erosion = cv2.erode(close, kernel)
dilate = cv2.dilate(close, kernel)
minLineLength = 1000
maxLineGap = 1

lines = cv2.HoughLinesP(open,1,np.pi/180, 150, minLineLength, maxLineGap)
for line in lines:
    x1, y1, x2, y2 = line[0][0],line[0][1],line[0][2],line[0][3]
    cv2.line(img1,(x1,y1),(x2,y2),(0,255,0),1)
cv2.imshow('lines', img1)
cv2.imshow('close', close)
cv2.imshow('threshold', th2)
cv2.imshow('open', open)
cv2.waitKey()
cv2.destroyAllWindows()

# img = cv2.imread('1.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray,50,150,apertureSize = 3)
# img = cv2.imread('1.jpg', 0)
# img1 = cv2.imread('1.jpg')
# # Otsu's thresholding
# ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# minLineLength = 100
# maxLineGap = 10
# lines = cv2.HoughLinesP(th2,1,np.pi/180,100,minLineLength,maxLineGap)
# for line in lines:
#     x1,y1,x2,y2 = line[0][0],line[0][1],line[0][2],line[0][3]
#     cv2.line(img1,(x1,y1),(x2,y2),(0,255,0),2)
# cv2.imshow('dd', th2)
# cv2.imshow('houghlines5.jpg',img1)
# cv2.waitKey()
# cv2.destroyAllWindows()
# img = cv2.imread('1.jpg', 0)
# img1 = cv2.imread('1.jpg')
# # Otsu's thresholding
# ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# lines = cv2.HoughLines(th2, 1, np.pi / 180, 300)  #这里对最后一个参数使用了经验型的值
# for line in lines:
#     rho = line[0][0]  #第一个元素是距离rho
#     theta = line[0][1]  #第二个元素是角度theta
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 500*(-b))
#     y1 = int(y0 + 500*(a))
#     x2 = int(x0 - 500*(-b))
#     y2 = int(y0 - 500*(a))
#     cv2.line(img1,(x1,y1),(x2,y2),(0,255,0),1)
# cv2.imshow('Result', img1)
# cv2.imshow('r', th2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = cv2.imread('3.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray, 50, 150, apertureSize=3)
# lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)  #这里对最后一个参数使用了经验型的值
# for line in lines:
#     rho = line[0][0]  #第一个元素是距离rho
#     theta = line[0][1]  #第二个元素是角度theta
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#     cv2.line(img,(x1,y1),(x2,y2),(5,255,0),2)
# cv2.imshow('Canny', edges)
# cv2.imshow('Result', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
