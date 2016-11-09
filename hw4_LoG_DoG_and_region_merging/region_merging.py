

#second_image_arr = cv2.imread('MixedVegetables.jpg',0)
#img2_row = len(second_image_arr) #268
#img2_col =  len(second_image_arr[0]) #400

#ret, thresh = cv2.threshold(second_image_arr,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#showImage(second_image_arr)
#showImage(thresh)

## noise removal
#kernel = np.ones((3,3),np.uint8)
#opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

#showImage(opening)
