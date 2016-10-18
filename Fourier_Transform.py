

import cv2
import numpy as np, sys
import scipy
import PIL
import math
import cmath
from scipy import misc 

##load image
#img = cv2.imread("file") --> loads image as an array.

img_Arr = cv2.imread("avengers.jpg")#load image
img_Arr = cv2.cvtColor(img_Arr, cv2.COLOR_RGB2GRAY) #convert img to grayscale


def showImage(img):
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#showImage(img_Arr)

def FT(img_A):
	rows = len(img_A)
	columns = len(img_A[0])
	
	for i in range(rows-1):
		for j in range(columns-1):
			y = -2*(math.pi)*1j*(float(i*i/rows) + float(j*j/columns))
			y_res = math.sqrt(y.real*y.real + y.imag*y.imag)/math.sqrt(rows*columns)
			result = math.pow(math.e,y_res)
			img_A[i][j] = np.uint8(result)
	showImage(img_Arr)
	return img_Arr

def inverseFT(img_A):
	rows = len(img_A)
	columns = len(img_A[0])
	for i in range(rows-1):
		for j in range(columns-1):
			y = cmath.phase(2*(math.pi)*1j*(float(i*i/rows) + float(j*j/columns)))
			y_res = math.sqrt(y.real*y.real + y.imag*y.imag)/math.sqrt(rows*columns)
			result = math.pow(math.e,y_res)
			img_A[i][j] = result
	showImage(img_Arr)
	return img_Arr
ft = FT(img_Arr)
inv_ft = inverseFT(ft)
