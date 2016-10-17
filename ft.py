
import cv2
import numpy as np, sys
import scipy
import PIL
from scipy import misc 
from scipy.ndimage.filters import gaussian_filter


##load image
#img = cv2.imread("file") --> loads image as an array.

img_Arr = cv2.imread("blackwidow.jpg")#load image
img_Arr = cv2.cvtColor(img_Arr, cv2.COLOR_RGB2GRAY) #convert img to grayscale

def showImage(img):
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
#showImage(img_Arr)
gaussian_filter = np.array([[1.0,2.0,1.0],[2.0,4.0,2.0],[1.0,2.0,1.0]], dtype = np.float) #gaussian filter matrix

print gaussian_filter

#rows = len(img_Arr) #480 px
#columns = len(img_Arr[0]) #320 px
#gaussian_result = np.ones(shape = (rows,columns),dtype = np.float)

#darkening the image
def darken(img_Arr):
	for i in range(rows):
		for j in range(columns):
			val = img_Arr[i][j] / 4
			img_Arr[i][j] = val

#showImage(img_Arr)
#darken(img_Arr)
#showImage(img_Arr)
#result = np.dot(img_Arr,gaussian_filter)

#downSize will return the result array(downsized array).
def downSize(img_A):
	
	#downsize by 1/4 because 1/2 in row and 1/2 in column
	result = img_A[::2,::2] 
	res = np.uint8(result)
	#result = np.ones(shape = (rows/2+1,columns/2+1),dtype=np.int)
	#np_i =0
	#for i in range(rows):
	#	if(i%2==0):
	#		np_j=0
	#		if(j%2==0):
	#			result[np_i][np_j] = img_Arr[i][j]
	#			np_j +=1
	#	np_i +=1
	#print result
	
	showImage(res)
	return res

#gaussianBlur will return the result array
def gaussianBlur(img_A,gf):
	#blurred = gaussian_filter(img_Arr, sigma=7)
	#showImage(blurred)
	result = img_A.astype(float)
	rows = len(img_A)
	columns = len(img_A[0])
	for i in range(rows):
		for j in range(columns):
			if(i==0 or j ==0 or i == rows-1 or j == columns -1):
				result[i][j] = float(0)
			else:
				#multiply original image x kernel
				val = img_A[i-1][j-1]*gf[0][0] + img_A[i][j-1]*gf[1][0] + img_A[i+1][j-1]*gf[2][0] + img_A[i-1][j]*gf[0][1] + img_A[i][j]*gf[1][1] + img_A[i+1][j]*gf[2][1] + img_A[i-1][j+1]*gf[0][2] + img_A[i][j+1]*gf[1][2] + img_A[i+1][j+1]*gf[2][2]
				result[i][j] = float(val / 16.0)
	result *= 255.0/result.max()
	res = np.uint8(result)
	showImage(res)
	return res

gaussianBlur(downSize(gaussianBlur(downSize(gaussianBlur(img_Arr,gaussian_filter)),gaussian_filter)),gaussian_filter)
#gaussianBlur(gaussian_result,gaussian_filter,gaussian_result)
#gaussianBlur(gaussian_result,gaussian_filter,gaussian_result)

