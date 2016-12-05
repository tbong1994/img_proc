
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

ground_img1 = cv2.imread('disp1.png',0) #compare disparity map result
ground_img2 = cv2.imread('disp5.png',0)
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
#Then try to find the most identical block from the other images. Ideally SSD should be 0.

#First procedure is to set the block 3x3 and 9x9 for this project.
#iterate through the input array and for each 3x3 or 9x9 block, try searching for the most
#identical block in the other array, starting from the same coordinates and setting
#the maximum horizontal distance to search for. Only need to search in the x axis because
#All three images have the same height and level. ==> This is RECTIFICATION.

#USE the Sum of Squared Differences. ==> for each block to compare, do the following:
#1)Calcalate difference of each pixel from image1 to image2 then square the difference.
#2)After all the pixels within the block have been calculated, add all the difference values together.
#3)The added value is the determinant of similarities(SSD). The lower the value == the closer it is.
#4)The smallest SSD indicates the most similar block. Calculate the column distance from img1 to the index of the pixel value with the lowest SSD value from img2.

##ITERATING THROUGH EVERY ELEMENT IS INEFFICIENT BECAUSE YOU KNOW THE IMAGES ONLY DIFFER SLIGHTLY IN POSITION.
##SET THE DISTANCE SEARCH AND ONLY ITERATE THROUGH THAT DOMAIN SEARCH, ie) 10 or 20 px to the left and to the right.


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
	
	for i in range(1, row-1): #3x3 only able from (1,1) and ends when (last-1,last-1)
		for j in range(1, col-1):
			block = np.array([(img1[i-1][j-1],img1[i-1][j],img1[i-1][j+1]),
							(img1[i][j-1],img1[i][j],img1[i][j+1]),
							(img1[i+1][j-1],img1[i+1][j],img1[i+1][j+1])])
				
			####I'M GOING TO SET THE SEARCH DISTANCE TO 20. SO 10 TO THE LEFT AND 10 TO THE RIGHT.
			##MAKE THIS FUNCTION WORK FOR BOTH IMAGES.. SO IT NEEDS TO ITERATE 20 PIXELS SUCCESSFULLY WHEREVER THE INDEX IS..
			
			if(j<50):
				ssd_min = 100
				for k in range(1,j+50): #j is close to the left most side.. iterate to the end of the left side and iterate 10 to the right.
					ssd = math.pow(block[0][0]-img2[i-1][k-1],2)+ math.pow(block[0][1]-img2[i-1][k],2)
					+math.pow(block[0][2]-img2[i-1][k+1],2) + math.pow(block[1][0]-img2[i][k-1],2)
					+math.pow(block[1][1]-img2[i][k],2) + math.pow(block[1][2]-img2[i][k+1],2)
					+math.pow(block[2][0]-img2[i+1][k-1],2) + math.pow(block[2][1]-img2[i+1][k],2)
					+math.pow(block[2][2]-img2[i+1][k+1],2)
						
					if(ssd < ssd_min): #found the matching block.
						#get the distance of pixels.
						dist = j - k # j is x index of orig image, k is x index of 2nd image.
						disparity_values[i][j] = dist
			
			if(col-1-j<50): #j is close to the right most side.. iterate to the end of the right side and iterate 10 to the left.
				ssd_min = 100
				for k in range(j-50, col-1):
					ssd = math.pow(block[0][0]-img2[i-1][k-1],2)+ math.pow(block[0][1]-img2[i-1][k],2)
					+math.pow(block[0][2]-img2[i-1][k+1],2) + math.pow(block[1][0]-img2[i][k-1],2)
					+math.pow(block[1][1]-img2[i][k],2) + math.pow(block[1][2]-img2[i][k+1],2)
					+math.pow(block[2][0]-img2[i+1][k-1],2) + math.pow(block[2][1]-img2[i+1][k],2)
					+math.pow(block[2][2]-img2[i+1][k+1],2)
				
					if(ssd < ssd_min): #found the matching block.
						#get the distance of pixels.
						dist = j - k # j is x index of orig image, k is x index of 2nd image.
						disparity_values[i][j] = dist
					#if(ssd < ssd_min): #replace lowest ssd value.
						#ssd_min = ssd
				#ssd_values[i][j] = ssd_min
			
			else: #everywhere else. iterate 10 to the left and 10 to the right.
				ssd_min = 100
				for k in range(j-50,j+50): #only calcaulte in the x direction.
					ssd = math.pow(block[0][0]-img2[i-1][k-1],2)+ math.pow(block[0][1]-img2[i-1][k],2)
					+math.pow(block[0][2]-img2[i-1][k+1],2) + math.pow(block[1][0]-img2[i][k-1],2)
					+math.pow(block[1][1]-img2[i][k],2) + math.pow(block[1][2]-img2[i][k+1],2)
					+math.pow(block[2][0]-img2[i+1][k-1],2) + math.pow(block[2][1]-img2[i+1][k],2)
					+math.pow(block[2][2]-img2[i+1][k+1],2)
										
					if(ssd < ssd_min): #found the matching block.
						#get the distance of pixels.
						dist = j - k # j is x index of orig image, k is x index of 2nd image.
						disparity_values[i][j] = dist
					#if(ssd < ssd_min): #replace lowest ssd value.
						#ssd_min = ssd
				#ssd_values[i][j] = ssd_min
				
	showImage(disparity_values)
	#showImage(np.uint(disparity_values))
	#print disparity_values
	
	return disparity_values



def block_matching_9(img1,img2):
	#3x3 block matching
	row = len(img1)
	col = len(img1[0])
	#print len(ssd_values)
	#print len(ssd_values[0])
	
	disparity_values = np.zeros(shape = (row,col),dtype=np.float)#keep track of SSD values at that index.
	dist = 0 #distance value of 2 pix
	
	for i in range(4,row-4): #9x9 only able from (4,4) and ends when (last-4,last-4)
		for j in range(4,col-4):
			#set block 9x9
			block = np.array([(img1[i-4][j-4],img1[i-4][j-3],img1[i-4][j-2],img1[i-4][j-1],img1[i-4][j],img1[i-4][j+1],img1[i-4][j+2],img1[i-4][j+3],img1[i-4][j+4]),
							(img1[i-3][j-4],img1[i-3][j-3],img1[i-3][j-2],img1[i-3][j-1],img1[i-3][j],img1[i-3][j+1],img1[i-3][j+2],img1[i-3][j+3],img1[i-3][j+4]),
							(img1[i-2][j-4],img1[i-2][j-3],img1[i-2][j-2],img1[i-2][j-1],img1[i-2][j],img1[i-2][j+1],img1[i-2][j+2],img1[i-2][j+3],img1[i-2][j+4]),
							(img1[i-1][j-4],img1[i-1][j-3],img1[i-1][j-2],img1[i-1][j-1],img1[i-1][j],img1[i-1][j+1],img1[i-1][j+2],img1[i-1][j+3],img1[i-1][j+4]),
							(img1[i][j-4],img1[i][j-3],img1[i][j-2],img1[i][j-1],img1[i][j],img1[i][j+1],img1[i][j+2],img1[i][j+3],img1[i][j+4]),
							(img1[i+1][j-4],img1[i+1][j-3],img1[i+1][j-2],img1[i+1][j-1],img1[i+1][j],img1[i+1][j+1],img1[i+1][j+2],img1[i+1][j+3],img1[i+1][j+4]),
							(img1[i+2][j-4],img1[i+2][j-3],img1[i+2][j-2],img1[i+2][j-1],img1[i+2][j],img1[i+2][j+1],img1[i+2][j+2],img1[i+2][j+3],img1[i+2][j+4]),
							(img1[i+3][j-4],img1[i+3][j-3],img1[i+3][j-2],img1[i+3][j-1],img1[i+3][j],img1[i+3][j+1],img1[i+3][j+2],img1[i+3][j+3],img1[i+3][j+4]),
							(img1[i+4][j-4],img1[i+4][j-3],img1[i+4][j-2],img1[i+4][j-1],img1[i+4][j],img1[i+4][j+1],img1[i+4][j+2],img1[i+4][j+3],img1[i+4][j+4]),])
			
			####I'M GOING TO SET THE SEARCH DISTANCE TO 10. SO 5 TO THE LEFT AND 5 TO THE RIGHT.
			##MAKE THIS FUNCTION WORK FOR BOTH IMAGES.. SO IT NEEDS TO ITERATE 10 PIXELS SUCCESSFULLY WHEREVER THE INDEX IS..
			
			if(j<30):
				for k in range(4,j+30):
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
			if(col-1-j<30): 
				for k in range(j-30, col-4):
					ssd = math.pow(block[0][0]-img2[i-1][k-1],2)+ math.pow(block[0][1]-img2[i-1][k],2)
					+math.pow(block[0][2]-img2[i-1][k+1],2) + math.pow(block[1][0]-img2[i][k-1],2)
					+math.pow(block[1][1]-img2[i][k],2) + math.pow(block[1][2]-img2[i][k+1],2)
					+math.pow(block[2][0]-img2[i+1][k-1],2) + math.pow(block[2][1]-img2[i+1][k],2)
					+math.pow(block[2][2]-img2[i+1][k+1],2)
				
					
					if(ssd < 100): #least SSD
						#get the distance of pixels.
						dist = j - k # j is x index of orig image, k is x index of 2nd image.
						disparity_values[i][j] = dist
			else: #everywhere else
				for k in range(j-30,j+30): #only calcaulte in the x direction.
					ssd = math.pow(block[0][0]-img2[i-1][k-1],2)+ math.pow(block[0][1]-img2[i-1][k],2)
					+math.pow(block[0][2]-img2[i-1][k+1],2) + math.pow(block[1][0]-img2[i][k-1],2)
					+math.pow(block[1][1]-img2[i][k],2) + math.pow(block[1][2]-img2[i][k+1],2)
					+math.pow(block[2][0]-img2[i+1][k-1],2) + math.pow(block[2][1]-img2[i+1][k],2)
					+math.pow(block[2][2]-img2[i+1][k+1],2)
										
					if(ssd < 100): #found the matching block.
						#get the distance of pixels.
						dist = j - k # j is x index of orig image, k is x index of 2nd image.
						disparity_values[i][j] = dist
					
	showImage(disparity_values)
	showImage(np.uint(disparity_values))
	
	return disparity_values

def dynamic_disp(img1,img2):
	rows = len(img1)
	columns = len(img1[0])
	
	result = np.zeros(shape = (rows,columns),dtype=np.float) #output array
	out_row = 0 #output array index row
	
	##get the longest common subsequent between 2 images. occlusions are blank(0).? 
	
	#FOR EACH ROW(1 FROM IMG1, 1 FROM IMG2), YOU NEED AN INDEX. say i is index for img1 and j is index for img2. if elem[i] == elem[j] then append the element to output array and i++, j++.
	#if elem[i] != elem[j] then only i++ or j++(doesn't matter which one). Keep going until either i or j == last index. Repeat for each row.
	
	img2_row = 0 #index for img2 row
	img1_row = 0
	#count = 0
	
	sd = 20 #search distance
	for i in range(rows):
		for j in range(columns):
			img1_values = img1[i][j:j+sd]
			img2_values ] img2[i][j:j+sd]
			lcs_values = lcs(img1_values,img2_values,result)
			if(len(img1[0] - j - 1 < 20):
				j+= 20 #skip to next block to find the lcs.
			else:
				j+= len(img1[0] - j - 1
				
	#print img2_row #370
	#print count
	
	#print result
	
	#showImage(np.uint(result))
	#showImage(result)

def lcs(X, Y,result):
	m = len(X) #length of col
	n = len(Y)
	
	L = [[0 for x in xrange(n+1)] for x in xrange(m+1)]
 
    # Following steps build L[m+1][n+1] in bottom up fashion. Note
    # that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1] 
	for i in xrange(m+1):
		for j in xrange(n+1):
			if i == 0 or j == 0:
				L[i][j] = 0
			elif X[i-1] == Y[j-1]:
				L[i][j] = L[i-1][j-1] + 1
			else:
				L[i][j] = max(L[i-1][j], L[i][j-1])
 
    # Following code is used to print LCS
	index = L[m][n]
	
	# Create a character array to store the lcs string
	lcs = [""] * (index+1)
	lcs[index] = "\0"
	
	# Start from the right-most-bottom-most corner and
	# one by one store characters in lcs[]
	i = m
	j = n
	while i > 0 and j > 0:
	
		# If current character in X[] and Y are same, then
		# current character is part of LCS
		if X[i-1] == Y[j-1]:
			lcs[index-1] = X[i-1]
			i-=1
			j-=1
			index-=1
	
		# If not same, then find the larger of two and
		# go in the direction of larger value
		elif L[i-1][j] > L[i][j-1]:
			i-=1
		else:
			j-=1

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

##DISPARITY WITH BLOCK 9X9

#disp2 = block_matching_9(img1,img2)

##DISPARITY WITH DYNAMIC PROGRAMMING
#disp3 = dynamic_disp(img2,img1)

##MSE CALCULATION. COMPARE WITH THE PROVIDED IMAGES

#mse1 = computeMSE(disp1,ground_img1)
#mse2 = computeMSE(disp1,ground_img2)

#mse3 = computeMSE(disp2,ground_img1)
#mse4 = computeMSE(disp2,ground_img2)

#mse5 = computeMSE(disp3,ground_img1)
#mse6 = computeMSE(disp3,ground_img2)
