import cv2
import numpy as np, sys
import scipy
import PIL
from scipy import misc 
import math

##load image
#img = cv2.imread("file") --> loads image as an array.

img_Arr = cv2.imread("avengers.jpg")#load image
img_Arr = cv2.cvtColor(img_Arr, cv2.COLOR_RGB2GRAY) #convert img to grayscale

def showImage(img):
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
#showImage(img_Arr)
gaussian_filter = np.array([[1.0,2.0,1.0],[2.0,4.0,2.0],[1.0,2.0,1.0]], dtype = np.float) #gaussian filter matrix

showImage(img_Arr)
rows = len(img_Arr) #392 px
columns = len(img_Arr[0]) #597 px

#global output array
#res = img_Arr

#downSize will return the result array(downsized array).
def downSize(img_A):
	rows = len(img_A)
	columns = len(img_A[0])
	result = img_A[::2,::2]
	#result = np.ones(shape = (rows/2,columns/2+1),dtype=np.int)
	#np_i =0
	#for i in range(rows):
	#	if(i%2==0):
	#		np_j=0
	#		for j in range(columns):
	#			if(j%2==0):
	#				result[np_i][np_j] = img_A[i][j]
	#				np_j +=1
	#		np_i +=1
	res = np.uint8(result)
	#showImage(res)
	return res
	
def enlargeSize(img_A):
	#result = scipy.misc.imresize(img_A, 200)
	rows = len(img_A)*2
	columns = len(img_A[0])*2
	result = np.ones(shape = (rows,columns),dtype=np.int)
	np_i = 0 #index for the input array.
	for i in range(rows):
		np_j = 0
		for j in range(columns):
			if(i%2==0 and j % 2 ==0):
				result[i][j] = img_A[np_i][np_j]
				result[i+1][j] = img_A[np_i][np_j]
				result[i][j+1] = img_A[np_i][np_j]
				result[i+1][j+1] = img_A[np_i][np_j]
				np_j += 1
		if(i==rows):
			for j in range(columns):
				result[i][j] = img_A[np_i][np_j]
				result[i+1][j] = img_A[np_i][np_j]
				result[i][j+1] = img_A[np_i][np_j]
				result[i+1][j+1] = img_A[np_i][np_j]
		if(i%2 ==0 and i != 0):
			np_i +=1
	res = np.uint8(result)
	##showImage(res)
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
	#showImage(res)
	return res

#subtract the G1 with G1'(blurred/enlarged image).
def laPlacian(original, enlarged): #try averaging out the extra row with adjacent cells instead of having them all 1
	rows_O = len(original)
	columns_O = len(original[0])
	rows_E = len(enlarged)
	columns_E = len(enlarged[0])

	if(rows_O != rows_E):
		if(rows_O > rows_E): #add a row to enlarged matrix
			for k in range(rows_O - rows_E):
				temp = np.ones(shape = (1, columns_E))
				for i in range(1):
					last_row_enlarged = rows_E-1
					for j in range(len(temp[0])):
						temp[i][j] = enlarged[last_row_enlarged][j]
				enlarged = np.concatenate((enlarged, temp),axis =0)
		else: #add a row to original matrix
			for k in range(rows_E - rows_O):
				temp = np.ones(shape = (1, columns_O))
				for i in range(1):
					last_row_original = rows_O-1
					for j in range(len(temp[0])):
						temp[i][j] = enlarged[last_row_original][j]
				enlarged = np.concatenate((enlarged, temp),axis =0)
	if(columns_O != columns_E):
		if(columns_O > columns_E): #add a column to enlarged matrix
			for k in range(columns_O - columns_E):	
				temp = np.ones(shape = (rows_E, 1))
				for i in range(len(temp)):
					last_column_enlarged = columns_E-1
					for j in range(1):
						temp[i][j] = enlarged[i][last_column_enlarged]
				enlarged = np.concatenate((enlarged, temp),axis =1)
		else: #add a column to original matrix
			for k in range(columns_E - columns_O):
				temp = np.ones(shape = (rows_O, 1))
				for i in range(len(temp)):
					last_column_original = columns_O-1
					for j in range(1):
						temp[i][j] = original[i][last_column_original]
				original = np.concatenate((original, temp),axis =1)				
	result = np.subtract(original, enlarged)
	res = np.uint8(result)
	showImage(res)
	return res

def inv_Laplacian(original, enlarged):
	rows_O = len(original)
	columns_O = len(original[0])
	rows_E = len(enlarged)
	columns_E = len(enlarged[0])
	if(rows_O != rows_E):
		if(rows_O > rows_E): #add a row to enlarged matrix
			for k in range(rows_O - rows_E):
				temp = np.ones(shape = (1, columns_E))
				for i in range(1):
					last_row_enlarged = rows_E-1
					for j in range(len(temp[0])):
						temp[i][j] = enlarged[last_row_enlarged][j]
				enlarged = np.concatenate((enlarged, temp),axis =0)
		else: #add a row to original matrix
			for k in range(rows_E - rows_O):
				temp = np.ones(shape = (1, columns_O))
				for i in range(1):
					last_row_original = rows_O-1
					for j in range(len(temp[0])):
						temp[i][j] = enlarged[last_row_original][j]
				print len(enlarged)
				print len(enlarged[0])
				print len(temp)
				print len(temp[0])
				enlarged = np.concatenate((enlarged, temp),axis =0)
	if(columns_O != columns_E):
		if(columns_O > columns_E): #add a column to enlarged matrix
			for k in range(columns_O - columns_E):	
				temp = np.ones(shape = (rows_E, 1))
				for i in range(len(temp)):
					last_column_enlarged = columns_E-1
					for j in range(1):
						temp[i][j] = enlarged[i][last_column_enlarged]
				enlarged = np.concatenate((enlarged, temp),axis =1)
		else: #add a column to original matrix
			for k in range(columns_E - columns_O):
				temp = np.ones(shape = (rows_O, 1))
				for i in range(len(temp)):
					last_column_original = columns_O-1
					for j in range(1):
						temp[i][j] = original[i][last_column_original]
				original = np.concatenate((original, temp),axis =1)
	result = np.add(original, enlarged)
	res = np.uint8(result)
	showImage(res)
	return res

def computeMSE(original, reconstructed):
	rows = len(original)
	columns = len(original[0])
	
	res = 0.0
	for i in range(rows):
		for j in range(columns):
			result = math.pow((original[i][j] - reconstructed[0][0]),2)
			res += result
	print res
	
g1 = gaussianBlur(img_Arr,gaussian_filter)
g2 = downSize(gaussianBlur(g1,gaussian_filter))
g3 = downSize(gaussianBlur(g2,gaussian_filter))
g4 = downSize(gaussianBlur(g3,gaussian_filter))
g5 = downSize(gaussianBlur(g4,gaussian_filter))
g6 = downSize(gaussianBlur(g5,gaussian_filter))

#pyramid #1
g1_prime = enlargeSize(g2)
l1 = laPlacian(g1,g1_prime)
#pyramid #2
g2_prime = enlargeSize(g3)
l2 = laPlacian(g2,g2_prime)
#pyramid #3
g3_prime = enlargeSize(g4)
l3 = laPlacian(g3,g3_prime)
#pyramid #4
g4_prime = enlargeSize(g5)
l4 = laPlacian(g4,g4_prime)
#pyramid #5
g5_prime = enlargeSize(g6)
l5 = laPlacian(g5,g5_prime)

#inverse pyramid 1
rev_l5 = enlargeSize(l5)
rev_l4 = inv_Laplacian(l4,rev_l5)
#inverse pyramid 2
rev_l4 = enlargeSize(l4)
rev_l3 = inv_Laplacian(l3,rev_l4)
#inverse pyramid 3
rev_l3 = enlargeSize(l3)
rev_l2 = inv_Laplacian(l2,rev_l3)
#inverse pyramid 4
rev_l2 = enlargeSize(l2)
rev_l1 = inv_Laplacian(l1,rev_l2)

rev_l0 = inv_Laplacian(img_Arr,rev_l1)

computeMSE(img_Arr, rev_l0) #the MSE value.
