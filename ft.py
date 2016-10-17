
import cv2
import numpy as np, sys
import scipy
from scipy import misc 

gaussian_result = []
##load image
#img = cv2.imread("file") --> loads image as an array.

img_Arr = cv2.imread("blackwidow.jpg")#load image
img_Arr = cv2.cvtColor(img_Arr, cv2.COLOR_RGB2GRAY) #convert img to grayscale

def showImage(img):
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
#showImage(img_Arr)
gaussian_filter = np.matrix([[1/16,2/16,1/16],[2/16,4/16,2/16],[1/16,2/16,1/16]]) #gaussian filter matrix

rows = len(img_Arr) #480 px
columns = len(img_Arr[0]) #320 px

#initialize 2d array.
for i in range(rows):
	gaussian_result.append([])
	for j in range(columns):
		gaussian_result[i].append(1)
		
#darkening the image
def darken(img_Arr):
	for i in range(rows):
		for j in range(columns):
			val = img_Arr[i][j] / 4
			img_Arr[i][j] = val

#showImage(img_Arr)
#darken(img_Arr)
showImage(img_Arr)
#result = np.dot(img_Arr,gaussian_filter)

def downSize(img_Arr):
	print img_Arr
	
	#downsize by 1/4 because 1/2 in row and 1/2 in column
	result = img_Arr[::2,::2] 
	
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
	
	showImage(result)
	print result

downSize(img_Arr)

def gaussianPyr(img_Arr,gf,result):
	for i in range(rows):
		for j in range(columns):
			if(i==0 or j ==0 or i == rows-1 or j == columns -1):
				result[i][j] = 0
			else:
				#multiply original image x kernel
				val = img_Arr[i-1][j-1]*gf[0][0] + img_Arr[i][j-1]*gf[1][0] + img_Arr[i+1][j-1]*gf[2][0] + img_Arr[i-1][j]*gf[0][1] +img_Arr[i][j]*gf[1][1] + img_Arr[i+1][j]*gf[2][1] + img_Arr[i-1][j+1]*gf[0][2] + img_Arr[i][j+1]*gf[1][2] + img_Arr[i+1][j+1]*gf[2][2]
				result[i][j] = val
#gaussianPyr(img_Arr,gaussian_filter,gaussian_result)

##gaussian operation..apply the kernel for each pixel(element) in the image's matrix
##then you use the dot product of the cell and then the sum of those multiplications
##will be the result for that index, add the result to the output array.
##laplacian is the difference between the gaussian of n and n+1
