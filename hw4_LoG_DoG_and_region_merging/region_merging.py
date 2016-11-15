import cv2
import numpy as np
np.seterr(over='ignore')
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
showImage(second_image_arr)
def histogram(img):
	hist = cv2.calcHist([img],[0],None,[256],[0,256])
	print hist

def crack_edge(img):
	#output array is twice the size of input.
	#extra rows/cols should be filled in with the difference of adjacent(4neighbor)cells
	#from the original input.
	row = len(img)
	col = len(img[0])
	
	result = np.zeros(shape = (row*2,col*2),dtype=np.float) #twice the row&col
	
	row_r = len(result)
	col_r = len(result[0])
	
	#print row_r #536
	#print col_r #800
	
	img_x = 0 #row index for original image
	
	for i in range(row_r):
		img_y = 0 # initialize img_y(original img column index)
		diff=0 #difference of adj pixels.
		#print i
		for j in range(col_r):
			#print j
			#CALCULATE THE NEIGHBOR DIFFERENCE HERE
			if(img_x==0): #nothing above.
				if(img_y ==0):#only right and below neighbors
					diff= img[img_x+1][img_y]-img[img_x][img_y+1]
				elif(img_y ==col-1):#only left and below neighbors
					diff = img[img_x][img_y-1]-img[img_x+1][img_y]
				else:#nothing above.
					diff= float(img[img_x][img_y-1]-img[img_x+1][img_y]-img[img_x][img_y+1])
			elif(img_x==row-1):#nothing below
				if(img_y==0):#only above and right neighbors
					diff= img[img_x-1][img_y]-img[img_x][img_y+1]
				elif(img_y ==col-1): #only left and above neighbors
					diff= img[img_x-1][img_y]-img[img_x][img_y-1]
				else:#nothing below
					diff= float(img[img_x-1][img_y]-img[img_x][img_y-1]-img[img_x][img_y+1])
			elif((img_y==0 and img_x!=0) or (img_y==0 and img_x!=row-1)):#when 1st or last col.
				if(img_y==0):#nothing to the left.
					diff= img[img_x-1][img_y]-img[img_x][img_y+1]-img[img_x+1][img_y]
				elif(img_y==col-1):
					diff= img[img_x-1][img_y]-img[img_x][img_y-1]-img[img_x+1][img_y]
			else: #everywhere else, so 4 neighbors
				diff = img[img_x][img_y-1]-img[img_x][img_y+1]-img[img_x-1][img_y]-img[img_x+1][img_y] #neighbor value difference.
			
			
			if(i%2==0 and j%2==0):
				result[i][j] = img[img_x][img_y] #copy from original image
				#print '%s%d' %("col: ",img_y)
			elif(i%2==0 and j%2==1 and img_y < col-2):
				#operate here and then increment img_y
				#need to fill in current index of new array and the cell below current index. ie) [x][y], [x-1][y]
				result[i][j] = diff
				result[i-1][j] = diff
				img_y +=1
			elif(i%2==1 and j%2==0):
				#assign the same value from x-1,j+1 or x,j+1
				result[i][j] = result[i-1][j+1]
			elif(i%2==1 and j == col_r-1 and img_x < row-1):
				#only increment img_x when all the columns have been visited.
				img_x +=1
	res = np.uint8(result)
	showImage(res)
	return np.uint8(result)

def threshold(img):
	thres = 180
	row = len(img)
	col = len(img[0])
	result = np.zeros(shape = (row,col),dtype=np.float)
	#result = cv2.Canny(img,50,200)
	
	for i in range(row):
		for j in range(col):
			if(img[i][j] < thres):
				result[i][j] = 0
			else:
				result[i][j] = img[i][j]
	result = np.uint8(result)		
	showImage(result)
	return result
	
def merge(img):
	#recursively check neighbor of a pixel. Check if abs(x - y) < T2, then merge. if not, don't merge.
	#HOW DO YOU MERGE THEM? ASSIGN THE VALUE AS CURRENT INDEX? 
	
ce = crack_edge(second_image_arr)
t1 = threshold(ce)
#ret, thresh = cv2.threshold(second_image_arr,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#showImage(second_image_arr)
#showImage(thresh)

# noise removal
#kernel = np.ones((3,3),np.uint8)
#opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

#showImage(opening)
#histogram(opening)
