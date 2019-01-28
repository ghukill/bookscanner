# bg crop

import cv2
import imutils
import numpy as np
# import pytesseract
from PIL import Image, ImageFilter
import matplotlib as plt


def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect


def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect


def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the warped image
	return warped


def crop_double_page(input_filepath, output_filepath, preview=False):

	# read image from filepath
	image = cv2.imread(input_filepath)

	ratio = image.shape[0] / 500.0
	orig = image.copy()
	image = imutils.resize(image, height=500)

	# https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html
	edged = cv2.Canny(
		image, # input image
		75, # minVal
		200 # maxVal
	)

	# https://docs.opencv.org/3.3.1/d4/d73/tutorial_py_contours_begin.html
	# opencv 3.1
	# img, cnts, hierarchy = cv2.findContours(
	# 	edged.copy(), # source image
	# 	cv2.RETR_LIST, # contour retrieval mode
	# 	cv2.CHAIN_APPROX_SIMPLE # contour approximation method
	# )
	# opencv 4.x
	cnts, hierarchy = cv2.findContours(
		edged.copy(), # source image
		cv2.RETR_LIST, # contour retrieval mode
		cv2.CHAIN_APPROX_SIMPLE # contour approximation method
	)

	# sort the contours
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		if len(approx) == 4:
			screenCnt = approx
			break

	warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

	# preview
	if preview:
		imgplot = plt.imshow(warped)

	# write
	else:
		cv2.imwrite(output_filepath, warped)


def sharpen_image(input_filepath, output_filepath):

	print('sharpening...')

	# open image
	img = Image.open(input_filepath)

	# sharpen
	img_sharp = img.filter(ImageFilter.SHARPEN)

	# write
	img_sharp.save(output_filepath)


