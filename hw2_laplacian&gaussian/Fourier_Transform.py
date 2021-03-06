
import cv2
import numpy as np, sys
import scipy
import PIL
import math
import cmath
from scipy import misc 

##load image
#img = cv2.imread("file") --> loads image as an array.

img_Arr = cv2.imread("img.jpg")#load image
img_Arr = cv2.cvtColor(img_Arr, cv2.COLOR_RGB2GRAY) #convert img to grayscale


def showImage(img):
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#showImage(img_Arr)

def FT(img_A):
	copy_Image = np.copy(img_A)
	result_Image = copy_Image.astype(complex)
	print result_Image.dtype
	rows = len(result_Image)
	columns = len(result_Image[0])
	for i in range(rows):
		for j in range(columns):
			result_Image [i][j] =  FT_Operation(result_Image,i,j) #FT on input image and gets the value from FT_Op
	res = np.uint8(result_Image)
	showImage(res)
	return res
	
def FT_Operation(img_A_Copy, ft_i, ft_j):
	rows = len(img_A_Copy)
	columns = len(img_A_Copy[0])
	for i in range(rows):
		for j in range(columns):
			y = -2j*(math.pi)*(float(i*ft_i/rows) + float(j*ft_j/columns))
			result = math.e**y
			result = result / (rows*columns)
	return result

def inverseFT(img_A):
	copy_Image = np.copy(img_A)
	rows = len(copy_Image)
	columns = len(copy_Image[0])
	result_Image = copy_Image.astype(float)
	for i in range(rows):
		for j in range(columns):
			result_Image [i][j] =  IFT_Operation(result_Image,i,j) 
	showImage(img_A)
	return img_A

def IFT_Operation(img_A_Copy, ft_i, ft_j):
	rows = len(img_A_Copy)
	columns = len(img_A_Copy[0])
	for i in range(rows):
		for j in range(columns):
			y = 2j*(math.pi)*(float(i*ft_i/rows) + float(j*ft_j/columns))
			result = math.e**y
			result = result/(rows*columns)
	return result
	
def computeMSE(original, reconstructed):
	rows = len(original)
	columns = len(original[0])
	
	res = 0.0
	for i in range(rows):
		for j in range(columns):
			result = math.pow((original[i][j] - reconstructed[i][j]),2)
			res += result
	print res

#ft = FT(img_Arr)
#ift = inverseFT(ft)
showImage(img_Arr)
ft = np.fft.fft2(img_Arr)
ft = np.uint8(ft)
showImage(ft)
ift = np.fft.ifft2(ft)
ift = np.uint8(ift)
showImage(ift)
computeMSE(ft,ift)
