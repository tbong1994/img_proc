
import cv2
import numpy as np, sys
import scipy
from scipy import misc 

##load image
#img = cv2.imread("file") --> loads image as an array.

img_Arr = cv2.imread("blackwidow.jpg")

def showImage(img):
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
img_Arr = cv2.cvtColor(img_Arr, cv2.COLOR_RGB2GRAY) #convert img to grayscale

print img_Arr
gaussian_filter = np.matrix([[1,2,1],[2,4,2],[1,2,1]])
showImage(img_Arr)

rows = len(img_Arr) #480
columns = len(img_Arr[0]) #320

#print "row: %d\tcolum: %d"%(rows,columns)

#blurring the image
for i in range(rows):
	for j in range(columns):
		val = img_Arr[i][j] / 9
		img_Arr[i][j] = val

showImage(img_Arr)
#result = np.dot(img_Arr,gaussian_filter)


##gaussian operation..apply the kernel for each pixel(element) in the image's matrix
##then you use the dot product of the cell and then the sum of those multiplications
##will be the result for that index, add the result to the output array.
##laplacian is the difference between the gaussian of n and n+1
