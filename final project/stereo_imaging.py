
#BLOCK MATCHING AND DYNAMIC PROGRAMMING
#http://mccormickml.com/2014/01/10/stereo-vision-tutorial-part-i/
import cv2
import numpy as np
np.seterr(over='ignore')
from scipy import signal
import PIL
import math
from matplotlib import pyplot as plt

img1 = cv2.imread('view3.png',0) #middle
img2 = cv2.imread('view5.png',0) #left most

row = len(img2)
col = len(img2[0])

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


#==============================================================================
#SSD just gives you the matching pixel in the other view. You will have to subtract the column values to obtain disparity at that pixel
#Example: If (300.20) in Left image found a match in (300,70) in right image -
#The disparity value at (300,20) for left image will be 70 - 20 = 50
#==============================================================================


def block_matching_3(img1,img2):
	#3x3 block matching
	row = len(img1)
	col = len(img1[0])
	#print len(ssd_values)
	#print len(ssd_values[0])
	
	disparity_values = np.zeros(shape = (row,col),dtype=np.float)#keep track of SSD values at that index.
	dist = 0 #distance value of 2 pix
	
	for i in range(row-1):
		for j in range(col-1):
			#set block 3x3
			if(i==0 or j==0): #3x3 only able from (1,1) and ends when (last-1,last-1)
				continue
			else:# 3x3 block ====WORKS===
				block = np.array([(img1[i-1][j-1],img1[i-1][j],img1[i-1][j+1]),
								(img1[i][j-1],img1[i][j],img1[i][j+1]),
								(img1[i+1][j-1],img1[i+1][j],img1[i+1][j+1])])
				#print block
			
			#calculate SSD here.
			
			#ssd_min = 10000000 #min ssd value
			
			####I'M GOING TO SET THE SEARCH DISTANCE TO 10. SO 5 TO THE LEFT AND 5 TO THE RIGHT.
			##MAKE THIS FUNCTION WORK FOR BOTH IMAGES.. SO IT NEEDS TO ITERATE 10 PIXELS SUCCESSFULLY WHEREVER THE INDEX IS..
			
			if(j<20 and j!=0):
				for k in range(1,j):
					ssd = math.pow(block[0][0]-img2[i-1][k-1],2)+ math.pow(block[0][1]-img2[i-1][k],2)
					+math.pow(block[0][2]-img2[i-1][k+1],2) + math.pow(block[1][0]-img2[i][k-1],2)
					+math.pow(block[1][1]-img2[i][k],2) + math.pow(block[1][2]-img2[i][k+1],2)
					+math.pow(block[2][0]-img2[i+1][k-1],2) + math.pow(block[2][1]-img2[i+1][k],2)
					+math.pow(block[2][2]-img2[i+1][k+1],2)
					
					#if(ssd < ssd_min): #replace lowest ssd value.
						#ssd_min = ssd
						
					if(ssd < 100): #found the matching block.
						#get the distance of pixels.
						dist = j - k # j is x index of orig image, k is x index of 2nd image.
						disparity_values[i][j] = dist
				#ssd_values[i][j] = ssd_min
			if(col-j<20):
				##ITERATE ONLY COL-J TIMES TO THE RIGHT IF COL-J <10
				for k in range(j, col-j):
					ssd = math.pow(block[0][0]-img2[i-1][k-1],2)+ math.pow(block[0][1]-img2[i-1][k],2)
					+math.pow(block[0][2]-img2[i-1][k+1],2) + math.pow(block[1][0]-img2[i][k-1],2)
					+math.pow(block[1][1]-img2[i][k],2) + math.pow(block[1][2]-img2[i][k+1],2)
					+math.pow(block[2][0]-img2[i+1][k-1],2) + math.pow(block[2][1]-img2[i+1][k],2)
					+math.pow(block[2][2]-img2[i+1][k+1],2)
				
					
					if(ssd < 100): #found the matching block.
						#get the distance of pixels.
						dist = j - k # j is x index of orig image, k is x index of 2nd image.
						disparity_values[i][j] = dist
					#if(ssd < ssd_min): #replace lowest ssd value.
						#ssd_min = ssd
				#ssd_values[i][j] = ssd_min
			else:
				##iterate from the left
				for k in range(j-20,j): #only calcaulte in the x direction.
					ssd = math.pow(block[0][0]-img2[i-1][k-1],2)+ math.pow(block[0][1]-img2[i-1][k],2)
					+math.pow(block[0][2]-img2[i-1][k+1],2) + math.pow(block[1][0]-img2[i][k-1],2)
					+math.pow(block[1][1]-img2[i][k],2) + math.pow(block[1][2]-img2[i][k+1],2)
					+math.pow(block[2][0]-img2[i+1][k-1],2) + math.pow(block[2][1]-img2[i+1][k],2)
					+math.pow(block[2][2]-img2[i+1][k+1],2)
										
					if(ssd < 100): #found the matching block.
						#get the distance of pixels.
						dist = j - k # j is x index of orig image, k is x index of 2nd image.
						disparity_values[i][j] = dist
					#if(ssd < ssd_min): #replace lowest ssd value.
						#ssd_min = ssd
				#ssd_values[i][j] = ssd_min
				##ITERATE TO THE RIGHT
				for k in range(j,j+19):
					ssd = math.pow(block[0][0]-img2[i-1][k-1],2)+ math.pow(block[0][1]-img2[i-1][k],2)
					+math.pow(block[0][2]-img2[i-1][k+1],2) + math.pow(block[1][0]-img2[i][k-1],2)
					+math.pow(block[1][1]-img2[i][k],2) + math.pow(block[1][2]-img2[i][k+1],2)
					+math.pow(block[2][0]-img2[i+1][k-1],2) + math.pow(block[2][1]-img2[i+1][k],2)
					+math.pow(block[2][2]-img2[i+1][k+1],2)
					
					if(ssd < 100): #found the matching block.
						#get the distance of pixels.
						dist = j - k # j is x index of orig image, k is x index of 2nd image.
						disparity_values[i][j] = dist
						
					#get the lowest SSD value and append that value to ssd values array.
					#if(ssd < ssd_min): #replace lowest ssd value.
						#ssd_min = ssd
				#ssd_values[i][j] = ssd_min
	#showImage(ssd_values)
	showImage(disparity_values)
	#showImage(np.uint(disparity_values))
	
	print disparity_values
	
	#return disparity_values



def block_matching_9(img1,img2):
	#3x3 block matching
	row = len(img1)
	col = len(img1[0])
	#print len(ssd_values)
	#print len(ssd_values[0])
	
	disparity_values = np.zeros(shape = (row,col),dtype=np.float)#keep track of SSD values at that index.
	dist = 0 #distance value of 2 pix
	
	for i in range(row-4):
		for j in range(col-4):
			#set block 9x9
			if(i<4 or j<4): #9x9 only able from (4,4) and ends when (last-4,last-4)
				continue
			else:# 9x9 block ====WORKS===
				block = np.array([(img1[i-4][j-4],img1[i-4][j-3],img1[i-4][j-2],img1[i-4][j-1],img1[i-4][j],img1[i-4][j+1],img1[i-4][j+2],img1[i-4][j+3],img1[i-4][j+4]),
								(img1[i-3][j-4],img1[i-3][j-3],img1[i-3][j-2],img1[i-3][j-1],img1[i-3][j],img1[i-3][j+1],img1[i-3][j+2],img1[i-3][j+3],img1[i-3][j+4]),
								(img1[i-2][j-4],img1[i-2][j-3],img1[i-2][j-2],img1[i-2][j-1],img1[i-2][j],img1[i-2][j+1],img1[i-2][j+2],img1[i-2][j+3],img1[i-2][j+4]),
								(img1[i-1][j-4],img1[i-1][j-3],img1[i-1][j-2],img1[i-1][j-1],img1[i-1][j],img1[i-1][j+1],img1[i-1][j+2],img1[i-1][j+3],img1[i-1][j+4]),
								(img1[i][j-4],img1[i][j-3],img1[i][j-2],img1[i][j-1],img1[i][j],img1[i][j+1],img1[i][j+2],img1[i][j+3],img1[i][j+4]),
								(img1[i+1][j-4],img1[i+1][j-3],img1[i+1][j-2],img1[i+1][j-1],img1[i+1][j],img1[i+1][j+1],img1[i+1][j+2],img1[i+1][j+3],img1[i+1][j+4]),
								(img1[i+2][j-4],img1[i+2][j-3],img1[i+2][j-2],img1[i+2][j-1],img1[i+2][j],img1[i+2][j+1],img1[i+2][j+2],img1[i+2][j+3],img1[i+2][j+4]),
								(img1[i+3][j-4],img1[i+3][j-3],img1[i+3][j-2],img1[i+3][j-1],img1[i+3][j],img1[i+3][j+1],img1[i+3][j+2],img1[i+3][j+3],img1[i+3][j+4]),
								(img1[i+4][j-4],img1[i+4][j-3],img1[i+4][j-2],img1[i+4][j-1],img1[i+4][j],img1[i+4][j+1],img1[i+4][j+2],img1[i+4][j+3],img1[i+4][j+4]),])
				#print block
			
			#ssd_min = 10000000 #min ssd value
			
			####I'M GOING TO SET THE SEARCH DISTANCE TO 10. SO 5 TO THE LEFT AND 5 TO THE RIGHT.
			##MAKE THIS FUNCTION WORK FOR BOTH IMAGES.. SO IT NEEDS TO ITERATE 10 PIXELS SUCCESSFULLY WHEREVER THE INDEX IS..
			
			if(j<20 and j!=0):
				for k in range(4,j):
					ssd = math.pow(block[0][0]-img2[i-4][k-4],2)+ math.pow(block[0][1]-img2[i-4][k-3],2)+ math.pow(block[0][2]-img2[i-4][k-2],2)+ math.pow(block[0][3]-img2[i-4][k-1],2)+ math.pow(block[0][4]-img2[i-4][k],2)+ math.pow(block[0][5]-img2[i-4][k+1],2)+ math.pow(block[0][6]-img2[i-4][k+2],2)+ math.pow(block[0][7]-img2[i-4][k+3],2)+ math.pow(block[0][8]-img2[i-4][k+4],2)
					+math.pow(block[1][0]-img2[i-3][k-4],2)+ math.pow(block[0][1]-img2[i-3][k-3],2)+ math.pow(block[0][2]-img2[i-3][k-2],2)+ math.pow(block[0][3]-img2[i-3][k-1],2)+ math.pow(block[0][4]-img2[i-3][k],2)+ math.pow(block[0][5]-img2[i-3][k+1],2)+ math.pow(block[0][6]-img2[i-3][k+2],2)+ math.pow(block[0][7]-img2[i-3][k+3],2)+ math.pow(block[0][8]-img2[i-3][k+4],2)
					+math.pow(block[2][0]-img2[i-2][k-4],2)+ math.pow(block[0][1]-img2[i-2][k-3],2)+ math.pow(block[0][2]-img2[i-2][k-2],2)+ math.pow(block[0][3]-img2[i-2][k-1],2)+ math.pow(block[0][4]-img2[i-2][k],2)+ math.pow(block[0][5]-img2[i-2][k+1],2)+ math.pow(block[0][6]-img2[i-2][k+2],2)+ math.pow(block[0][7]-img2[i-2][k+3],2)+ math.pow(block[0][8]-img2[i-2][k+4],2)
					+math.pow(block[3][0]-img2[i-1][k-4],2)+ math.pow(block[0][1]-img2[i-1][k-3],2)+ math.pow(block[0][2]-img2[i-1][k-2],2)+ math.pow(block[0][3]-img2[i-1][k-1],2)+ math.pow(block[0][4]-img2[i-1][k],2)+ math.pow(block[0][5]-img2[i-1][k+1],2)+ math.pow(block[0][6]-img2[i-1][k+2],2)+ math.pow(block[0][7]-img2[i-1][k+3],2)+ math.pow(block[0][8]-img2[i-1][k+4],2)
					+math.pow(block[4][0]-img2[i][k-4],2)+ math.pow(block[0][1]-img2[i][k-3],2)+ math.pow(block[0][2]-img2[i][k-2],2)+ math.pow(block[0][3]-img2[i][k-1],2)+ math.pow(block[0][4]-img2[i][k],2)+ math.pow(block[0][5]-img2[i][k+1],2)+ math.pow(block[0][6]-img2[i][k+2],2)+ math.pow(block[0][7]-img2[i][k+3],2)+ math.pow(block[0][8]-img2[i][k+4],2)
					+math.pow(block[5][0]-img2[i+1][k-4],2)+ math.pow(block[0][1]-img2[i+1][k-3],2)+ math.pow(block[0][2]-img2[i+1][k-2],2)+ math.pow(block[0][3]-img2[i+1][k-1],2)+ math.pow(block[0][4]-img2[i+1][k],2)+ math.pow(block[0][5]-img2[i+1][k+1],2)+ math.pow(block[0][6]-img2[i+1][k+2],2)+ math.pow(block[0][7]-img2[i+1][k+3],2)+ math.pow(block[0][8]-img2[i+1][k+4],2)
					+math.pow(block[6][0]-img2[i+2][k-4],2)+ math.pow(block[0][1]-img2[i+2][k-3],2)+ math.pow(block[0][2]-img2[i+2][k-2],2)+ math.pow(block[0][3]-img2[i+2][k-1],2)+ math.pow(block[0][4]-img2[i+2][k],2)+ math.pow(block[0][5]-img2[i+2][k+1],2)+ math.pow(block[0][6]-img2[i+2][k+2],2)+ math.pow(block[0][7]-img2[i+2][k+3],2)+ math.pow(block[0][8]-img2[i+2][k+4],2)
					+math.pow(block[7][0]-img2[i+3][k-4],2)+ math.pow(block[0][1]-img2[i+3][k-3],2)+ math.pow(block[0][2]-img2[i+3][k-2],2)+ math.pow(block[0][3]-img2[i+3][k-1],2)+ math.pow(block[0][4]-img2[i+3][k],2)+ math.pow(block[0][5]-img2[i+3][k+1],2)+ math.pow(block[0][6]-img2[i+3][k+2],2)+ math.pow(block[0][7]-img2[i+3][k+3],2)+ math.pow(block[0][8]-img2[i+3][k+4],2)
					+math.pow(block[8][0]-img2[i+4][k-4],2)+ math.pow(block[0][1]-img2[i+4][k-3],2)+ math.pow(block[0][2]-img2[i+4][k-2],2)+ math.pow(block[0][3]-img2[i+4][k-1],2)+ math.pow(block[0][4]-img2[i+4][k],2)+ math.pow(block[0][5]-img2[i+4][k+1],2)+ math.pow(block[0][6]-img2[i+4][k+2],2)+ math.pow(block[0][7]-img2[i+4][k+3],2)+ math.pow(block[0][8]-img2[i+4][k+4],2)
					
					#if(ssd < ssd_min): #replace lowest ssd value.
						#ssd_min = ssd
						
					if(ssd < 100): #found the matching block.
						#get the distance of pixels.
						dist = j - k # j is x index of orig image, k is x index of 2nd image.
						disparity_values[i][j] = dist
				#ssd_values[i][j] = ssd_min
			if(col-j<20):
				##ITERATE ONLY COL-J TIMES TO THE RIGHT IF COL-J <10
				for k in range(j, col-j):
					ssd = math.pow(block[0][0]-img2[i-1][k-1],2)+ math.pow(block[0][1]-img2[i-1][k],2)
					+math.pow(block[0][2]-img2[i-1][k+1],2) + math.pow(block[1][0]-img2[i][k-1],2)
					+math.pow(block[1][1]-img2[i][k],2) + math.pow(block[1][2]-img2[i][k+1],2)
					+math.pow(block[2][0]-img2[i+1][k-1],2) + math.pow(block[2][1]-img2[i+1][k],2)
					+math.pow(block[2][2]-img2[i+1][k+1],2)
				
					
					if(ssd < 100): #found the matching block.
						#get the distance of pixels.
						dist = j - k # j is x index of orig image, k is x index of 2nd image.
						disparity_values[i][j] = dist
					#if(ssd < ssd_min): #replace lowest ssd value.
						#ssd_min = ssd
				#ssd_values[i][j] = ssd_min
			else:
				##iterate from the left
				for k in range(j-20,j): #only calcaulte in the x direction.
					ssd = math.pow(block[0][0]-img2[i-1][k-1],2)+ math.pow(block[0][1]-img2[i-1][k],2)
					+math.pow(block[0][2]-img2[i-1][k+1],2) + math.pow(block[1][0]-img2[i][k-1],2)
					+math.pow(block[1][1]-img2[i][k],2) + math.pow(block[1][2]-img2[i][k+1],2)
					+math.pow(block[2][0]-img2[i+1][k-1],2) + math.pow(block[2][1]-img2[i+1][k],2)
					+math.pow(block[2][2]-img2[i+1][k+1],2)
										
					if(ssd < 100): #found the matching block.
						#get the distance of pixels.
						dist = j - k # j is x index of orig image, k is x index of 2nd image.
						disparity_values[i][j] = dist
					#if(ssd < ssd_min): #replace lowest ssd value.
						#ssd_min = ssd
				#ssd_values[i][j] = ssd_min
				##ITERATE TO THE RIGHT
				for k in range(j,j+19):
					ssd = math.pow(block[0][0]-img2[i-1][k-1],2)+ math.pow(block[0][1]-img2[i-1][k],2)
					+math.pow(block[0][2]-img2[i-1][k+1],2) + math.pow(block[1][0]-img2[i][k-1],2)
					+math.pow(block[1][1]-img2[i][k],2) + math.pow(block[1][2]-img2[i][k+1],2)
					+math.pow(block[2][0]-img2[i+1][k-1],2) + math.pow(block[2][1]-img2[i+1][k],2)
					+math.pow(block[2][2]-img2[i+1][k+1],2)
					
					if(ssd < 100): #found the matching block.
						#get the distance of pixels.
						dist = j - k # j is x index of orig image, k is x index of 2nd image.
						disparity_values[i][j] = dist
						
					#get the lowest SSD value and append that value to ssd values array.
					#if(ssd < ssd_min): #replace lowest ssd value.
						#ssd_min = ssd
				#ssd_values[i][j] = ssd_min
	#showImage(ssd_values)
	showImage(disparity_values)
	#showImage(np.uint(disparity_values))
	
	print disparity_values
	
	#return disparity_values
def dynamic_disp(img1,img2):
	rows = len(img1)
	columns = len(img1[0])
	
	result = np.ones(shape = (rows,columns),dtype=np.int) #output array
	
	##get the longest common subsequent between 2 images. occlusions are blank(0).? 
	
	
def computeMSE(img1, img2):
	rows = len(img1)
	columns = len(img1[0])
	
	res = 0.0
	for i in range(rows):
		for j in range(columns):
			result = math.pow((img1[i][j] - img2[i][j]),2)
			res += result
	print res

def validate_disparity(img1,img2):
	stereo = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET,ndisparities=16, SADWindowSize=15)
	disparity = stereo.compute(img1,img2)
	plt.imshow(disparity,'gray')
	plt.show()

#validate_disparity(img1,img2)
#validate_disparity(img2,img1)


##DISPARITY WITH BLOCK 3X3

#disp1 = block_matching_3(img1,img2)
#disp2 = block_matching_3(img2,img1)

##DISPARITY WITH BLOCK 9X9

#disp3 = block_matching_9(img1,img2)
#disp4 = block_matching_9(img2,img1)

##DISPARITY WITH DYNAMIC PROGRAMMING
