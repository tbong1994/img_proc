import cv2
import numpy as np
from scipy import signal
import PIL
import math
from matplotlib import pyplot as plt

second_image_arr = cv2.imread('MixedVegetables.jpg',0)

img2_row = len(second_image_arr) #268
img2_col =  len(second_image_arr[0]) #400

def showImage(img):
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return img

def histogram(img):
	hist = cv2.calcHist([img],[0],None,[256],[0,256])
	print hist

def crack_edge(img):
	#output array is twice the size of input.
	#extra rows/cols should be filled in with the difference of adjacent(4neighbor)cells
	#from the original input.
	row = len(img)
	col = len(img[0])
	
	result = np.zeros(shape = (row*2,col*2),dtype=np.int) #twice the row&col
	
	row_r = len(result)
	col_r = len(result[0])
	
	#print row_r #536
	#print col_r #800
	
	img_x = 0 #row index for original image
	
	for i in range(row_r):
		img_y = 0 # initialize img_y(original img column index)
		diff = 0 #initialize diff
		
		#CALCULATE THE NEIGHBOR DIFFERENCE HERE
		if(img_x==0 and img_y ==0): #only right and below neighbors
		if(img_x==0 and img_y ==col-1): #only left and below neighbors
		if(img_x==row-1 and img_y==0): #only above and right neighbors
		if(img_x==row-1 and img_y ==col-1): #only left and above neighbors
		else: #everywhere else, so 4 neighbors
			diff = img[img_x][img_y-1]-img[img_x][img_y+1]-img[img_x-1][img_y]-img[img_x+1][img_y] #neighbor value difference.
		
		for j in range(col_r):
			if(i%2==0 and j%2==0):
				result[i][j] = img[img_x][img_y] #copy from original image
				#print '%s%d' %("col: ",img_y)
			if(i%2==0 and j%2==1):
				#operate here and then increment img_y
				#need to fill in current index of new array and the cell below current index. ie) [x][y], [x-1][y]
				result[i][j] = diff
				result[i-1][j] = diff
				img_y +=1 
			if(i%2==1 and j%2==0):
				#operate here. should assign the same value for diff of neighbor values.
				result[i][j] = diff
			if(i%2==1 and j == col_r-1):
				#only increment img_x when all the columns have been visited.
				img_x +=1
	res = np.uint8(result)
	showImage(res)
	return np.uint8(result)
	
	
crack_edge(second_image_arr)

#ret, thresh = cv2.threshold(second_image_arr,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#showImage(second_image_arr)
#showImage(thresh)

# noise removal
#kernel = np.ones((3,3),np.uint8)
#opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

#showImage(opening)
#histogram(opening)
