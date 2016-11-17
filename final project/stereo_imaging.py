
#BLOCK MATCHING AND DYNAMIC PROGRAMMING
#http://mccormickml.com/2014/01/10/stereo-vision-tutorial-part-i/
import cv2
import numpy as np
np.seterr(over='ignore')
from scipy import signal
import PIL
import math
from matplotlib import pyplot as plt

img1 = cv2.imread('view1.png',0) #right most
img2 = cv2.imread('view3.png',0) #middle
img3 = cv2.imread('view5.png',0) #left most

row = len(img1)
col = len(img1[0])

#print row #370
#print col #463

def showImage(img):
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return img
	#print img.dtype

#BLOCK MATCHING is setting a size of 3x3 or 9x9 in this project, from one image.
#Then try to find the most identical block from the other images.

#First procedure is to set the block 3x3 and 9x9 for this project.
#iterate through the input array and for each 3x3 or 9x9 block, try searching for the most
#identical block in the other array, starting from the same coordinates and setting
#the maximum horizontal distance to search for. Only need to search in the x axis because
#All three images have the same height and level. ==> This is RECTIFICATION.

#USE Sum of Squared Differences. ==> for each block to compare, do the following:
#1)Calcalate difference of each pixel from image1 to image 2 then square the difference.
#3)After all the pixels within the block have been calculated, add all pixel values together.
#4)The added value is the determinant of similarities. The lower the value == the closer it is.

#YOU CAN HAVE AN ARRAY SIZE OF IMG/3 OR IMG/9, and each index represents the SSD value corresponding
#to the index from the image. Then when the array is all filled up, just get the lowest value and pick that.
#then initialize the array and move on to the next row.

#OR YOU CAN JUST ESTIMATE HOW FAR THE IMAGES HAVE BEEN SHIFTED, AND SET HOW FAR YOU WILL GO TO 
#SEARCH, BECAUSE THIS COULD SAVE SOME TIME.

def block_matching_3(img1,img2):
	#3x3 block matching
	row = len(img1)
	col = len(img1[0])
	#row_img1 = len(img1)/3
	#col_img1 = len(img1[0])/3
	ssd_values = np.zeros(shape = (row,col),dtype=np.float)#keep track of SSD values at that index.
	print len(ssd_values)
	print len(ssd_values[0])
	for i in range(row-1):
		for j in range(col-1):
			#set block 3x3
			if(i==0 or j==0): #3x3 only able from (1,1) and ends when (last-1,last-1)
				continue
			else:# 3x3 block
				block = np.array([(img1[i-1][j-1],img1[i-1][j],img1[i-1][j+1]),
								(img1[i][j-1],img1[i][j],img1[i][j+1]),
								(img1[i+1][j-1],img1[i+1][j],img1[i+1][j+1])])
			#calculate SSD here.
			img2_x = j #index for distance iterations in x directions in 2nd image.
			ssd_min = 0.0 #min ssd value
			if(col-j > 10):
				for k in range(10): #only calcaulte in the x direction.
					ssd = math.pow(block[0][0]-img2[i-1][k-1],2)+ math.pow(block[0][1]-img2[i-1][k],2)
					+math.pow(block[0][2]-img2[i-1][k+1],2) + math.pow(block[1][0]-img2[i][k-1],2)
					+math.pow(block[1][1]-img2[i][k],2) + math.pow(block[1][2]-img2[i][k+1],2)
					+math.pow(block[2][0]-img2[i+1][k-1],2) + math.pow(block[2][1]-img2[i+1][k],2)
					+math.pow(block[2][2]-img2[i+1][k+1],2)
					
					img2_x+=1 #increase index for x direction.
					
					#get the lowest SSD value and append that value to ssd values array.
					if(ssd_min==0.0): #for the initial ssd_min
						ssd_min = ssd
					elif(ssd < ssd_min): #replace lowest ssd value.
						ssd_min = ssd
			elif(col-j <10):
				for k in range(col-1):
					ssd = math.pow(block[0][0]-img2[i-1][k-1],2)+ math.pow(block[0][1]-img2[i-1][k],2)
					+math.pow(block[0][2]-img2[i-1][k+1],2) + math.pow(block[1][0]-img2[i][k-1],2)
					+math.pow(block[1][1]-img2[i][k],2) + math.pow(block[1][2]-img2[i][k+1],2)
					+math.pow(block[2][0]-img2[i+1][k-1],2) + math.pow(block[2][1]-img2[i+1][k],2)
					+math.pow(block[2][2]-img2[i+1][k+1],2)
					
					img2_x+=1 #increase index for x direction.
					#get the lowest SSD value and append that value to ssd values array.
					if(ssd_min==0.0): #for the initial ssd_min
						ssd_min = ssd
					elif(ssd < ssd_min): #replace lowest ssd value.
						ssd_min = ssd
			ssd_values[i][j] = ssd_min #save ssd value for index (i,j)
	print ssd_values
	#print len(ssd_values)
	#print len(ssd_values[0])
	showImage(np.uint8(ssd_values))
#img3 should be the input image.

block_matching_3(img3,img2)
block_matching_3(img3,img1)
block_matching_3(img2,img1)
	
