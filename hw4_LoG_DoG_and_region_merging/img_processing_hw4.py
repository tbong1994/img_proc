import cv2
import numpy as np
import scipy
import PIL
from scipy import misc 
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

	return cv2.filter2D(img,-1,result,mask) #return result

img_after = dog(first_image_arr,dog_mask)
print img_after
showImage(img_after)

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




#def zero_crossing(img):
	##look at four neighbors and if the sign changes, it's zero crossing.
	##if no zero crossing for index i, then fill the i with 0? 
	#for i in range(len(img)):
		#for j in range(len(img[0])):
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


second_image_arr = cv2.imread('MixedVegetables.jpg',0)
img2_row = len(second_image_arr) #268
img2_col =  len(second_image_arr[0]) #400

ret, thresh = cv2.threshold(second_image_arr,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
showImage(second_image_arr)
showImage(thresh)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

showImage(opening)
