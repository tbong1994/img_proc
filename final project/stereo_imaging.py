
#BLOCK MATCHING AND DYNAMIC PROGRAMMING
#http://mccormickml.com/2014/01/10/stereo-vision-tutorial-part-i/
import cv2
import numpy as np
from scipy import signal
import PIL
import math
from matplotlib import pyplot as plt

img1 = cv2.imread('view1.png',0)
img2 = cv2.imread('view3.png',0)
img3 = cv2.imread('view5.png',0)

row = len(img1)
col = len(img1[0])

print row #370
print col #463

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
#1)Square each pixel within the block, both image 1 and image 2
#2)Calcalate difference of each pixel from image1 to image 2
#3)After all the pixels within the block have been calculated, add all pixel values together.
#4)The added value is the determinant of similarities. The lower the value == the closer it is.

#YOU CAN HAVE AN ARRAY SIZE OF IMG/3 OR IMG/9, and each index represents the SSD value corresponding
#to the index from the image. Then when the array is all filled up, just get the lowest value and pick that.
#then initialize the array and move on to the next row.

#OR YOU CAN JUST ESTIMATE HOW FAR THE IMAGES HAVE BEEN SHIFTED, AND SET HOW FAR YOU WILL GO TO 
#SEARCH, BECAUSE THIS COULD SAVE SOME TIME.

def block_matching_3(img):
	#3x3 block matching
	row = len(img)/3
	col = len(img[0])/3
	
