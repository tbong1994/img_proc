import cv2
import numpy as np
import scipy
import PIL
from scipy import misc 
import math

first_image_arr = cv2.imread('UBCampus.jpg',0) #0 MEANS GRAYSCALE
second_image_arr = cv2.imread('MixedVegetables.jpg',0)

img1_row = len(first_image_arr) #329
img1_col =  len(first_image_arr[0]) #500

img2_row = len(second_image_arr) #268
img2_col =  len(second_image_arr[0]) #400

dog_mask = np.array([[0,0,-1,-1,-1,0,0],[0,-2,-3,-3,-3,-2,0],[-1,-3,5,5,5,-3,-1],
					[-1,-3,5,16,5,-3,-1],[-1,-3,5,5,5,-3,-1],[0,-2,-3,-3,-3,-2,0],
					[0,0,-1,-1,-1,0,0]], dtype = np.float) #7 x 7

log_mask = np.array([[0,0,1,0,0],[0,1,2,1,0],[1,2,-16,2,1],
					[0,1,2,1,0],[0,0,1,0,0]], dtype = np.float) #5 x 5

def showImage(img):
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
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
	#x iteration = img1.length-1-(img2.length/2)
	for i in range(img_row-1-(mask_row/2)):
		for j in range(img_col-1-(mask_col/2)):			
			#Figure out a function or something to calculate this.
			#If you write this whole thing down, it will be super long..
			#49 calculations man.. nope.
			
			#result[starting_x+i][starting_y+j] = 
			

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
	#x iteration = img1.length-1-(img2.length/2)
	for i in range(img_row-1-(mask_row/2)):
		for j in range(img_col-1-(mask_col/2)):			
			#Figure out a function or something to calculate this.
			#If you write this whole thing down, it will be super long..
			#49 calculations man.. nope.
			
			#result[starting_x+i][starting_y+j] = 


def zero_crossing(img):
	#look at four neighbors and if the sign changes, it's zero crossing.
	#if no zero crossing for index i, then fill the i with 0? 
	for i in range(len(img)):
		for j in range(len(img[0])):
			if(i==0 and j==0): #only 2 neighbors, right and below
			
			if(i==0 and j==len(img[0])-1): #(0,last) has 2 neighbors, left and below
				
			if(i==len(img)-1 and j==0): #(last,0) only has right and above.

			if(i==len(img)-1 and j==len(img[0])-1): #(last,last) only has left and above.
			
			else: #everywhere else.
				if(img[i][j]==0):
					#do nothing.
				else if(img[i][j]>0):#current index is positive.
					#check neighbors if negative from above, clockwise.
					if(img[i-1][j]<0):
						#assign some value to output array.
					if(img[i][j+1]<0):
					if(img[i+1][j]<0):
					if(img[i][j-1]<0):
				else if(img[i][j]<0):#current index is negative.
					if(img[i-1][j]>0):
					if(img[i][j+1]>0):
					if(img[i+1][j]>0):
					if(img[i][j-1]>0):
	#sign change detecting function.
	asign = np.sign(img)
	signchange = ((np.roll(asign, 1) - asign) != 0).astype(uint8)
	
	#avoid 0's from having its own sign.
	zero = asign ==0
	while(zero.any()):
		asign[zero] = np.roll(asign, 1)[zero]
		zero = asign ==0
		
	signchange[0] = 0 #avoid circular detection. if not avoided, index 0 and last index will be compared. I don't want that.
	print signchange
				
