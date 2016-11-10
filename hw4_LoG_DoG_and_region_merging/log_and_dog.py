import cv2
import numpy as np
from scipy import signal
import PIL
import math

first_image_arr = cv2.imread('UBCampus.jpg',0) #0 MEANS GRAYSCALE



img1_row = len(first_image_arr) #329
img1_col =  len(first_image_arr[0]) #500

dog_mask = np.array([[0,0,-1,-1,-1,0,0],[0,-2,-3,-3,-3,-2,0],[-1,-3,5,5,5,-3,-1],
					[-1,-3,5,16,5,-3,-1],[-1,-3,5,5,5,-3,-1],[0,-2,-3,-3,-3,-2,0],
					[0,0,-1,-1,-1,0,0]], dtype = np.float) #7 x 7
log_mask = np.array([[0,0,1,0,0],[0,1,2,1,0],[1,2,-16,2,1],
					[0,1,2,1,0],[0,0,1,0,0]], dtype = np.float) #5 x 5

def showImage(img):
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return img
	#print img.dtype
	
##USE CONCOLVE 2D PYTHON LIBRARY FOR FILTERING. OPENCV CONVENTION IS WEIRD, THAT'S WHY U ONLY GET 255'S
def dog(img,mask):
	#calculate.
	#starting index.x = img.x/mask.x 
	#starting index.y = img.y/mask.y
	img_row = len(img)
	img_col = len(img[0])
	mask_row = len(mask)
	mask_col = len(mask[0])
	
	starting_x = img_row/mask_row
	starting_y = img_col/mask_col
	
	result = np.ones(shape = (img_row,img_col),dtype=np.int) #output array

	#return cv2.filter2D(img,-1,result,mask) #return result
	return signal.convolve2d(img, mask, boundary='symm', mode='same')

img_after_dog = dog(first_image_arr,dog_mask)
#showImage(img_after_dog)

def log(img,mask):
	#calculate.
	#starting index.x = img.x/mask.x 
	#starting index.y = img.y/mask.y
	img_row = len(img)
	img_col = len(img[0])
	mask_row = len(mask)
	mask_col = len(mask[0])
	
	starting_x = img_row/mask_row
	starting_y = img_col/mask_col
	
	result = np.ones(shape = (img_row,img_col),dtype=np.int) #output array
	return signal.convolve2d(img, mask, boundary='symm', mode='same')

img_after_log = log(first_image_arr,dog_mask)
#showImage(img_after_log)
#print(img_after_log)

## USE OUTPUT ARRAY AND MAKE A BINARY IMAGE== BLACK OR WHITE WHENEVER ZERO CROSSING OCCURS.

def zero_crossing(img):
	##look at four neighbors and if the sign changes, it's zero crossing.
	row = len(img)
	col = len(img[0])
	result = np.zeros(shape = (row,col),dtype=np.int)
	#print result
	
	asign = np.sign(img)
	sz = asign == 0
	while sz.any():
	    asign[sz] = np.roll(asign, 1)[sz]
	    sz = asign == 0
	signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
	signchange[0]=0
	for i in range(len(signchange)):
		for j in range(len(signchange[0])):
			if(signchange[i][j]==1):
				signchange[i][j]= 255
			else:
				signchange[i][j]=0
	res = np.uint8(signchange)
	showImage(res)
	return res
	
	#for i in range(row):
		#for j in range(col):
			#if(i==0 and j==0): #only 2 neighbors, right and below
			#if(i==0 and j==len(img[0])-1): #(0,last) has 2 neighbors, left and below
				
			#if(i==len(img)-1 and j==0): #(last,0) only has right and above.

			#if(i==len(img)-1 and j==len(img[0])-1): #(last,last) only has left and above.
			
			#else: #everywhere else.
				#if(img[i][j]==0):
					##do nothing.
				#else if(img[i][j]>0):#current index is positive.
					##check neighbors if negative from above, clockwise.
					#if(img[i-1][j]<0):
						##assign some value to output array.
					#if(img[i][j+1]<0):
					#if(img[i+1][j]<0):
					#if(img[i][j-1]<0):
				#else if(img[i][j]<0):#current index is negative.
					#if(img[i-1][j]>0):
					#if(img[i][j+1]>0):
					#if(img[i+1][j]>0):
					#if(img[i][j-1]>0):
def rid_weak_edge(img):
	hist = cv2.calcHist([img],[0],None,[256],[0,256])

	result = cv2.Canny(img,0,0)
	showImage(result)
	return result
	
first_zc = zero_crossing(img_after_log)
rid_weak_edge(first_zc)
