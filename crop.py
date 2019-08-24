# crop

import cv2
from distutils.version import StrictVersion
import imutils
import logging
import matplotlib as plt
import numpy as np
from PIL import Image
import re


# setup logger
logging.basicConfig(level=logging.DEBUG)
# parso shims
logging.getLogger('parso.python.diff').disabled = True
logging.getLogger('parso.cache').disabled = True
logger = logging.getLogger(__name__)


def order_points(pts):

	'''
	https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
	'''

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

	'''
	https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
	'''

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


def crop_and_rotate(
	input_filepath,
	output_filepath,
	preview=False):

	'''
	Crop away black background, transforming and rotating as well

	Looking for contours in this range:
	169700.0
	169688.0
	17359.0
	17356.0
	15537.5

	These are too small:
	4747.0
	4710.0
	2130.5
	1864.0
	581.0
	'''

	# read image from filepath
	image = cv2.imread(input_filepath)

	# resize
	ratio = image.shape[0] / 500.0
	orig = image.copy()
	image = imutils.resize(image, height=500)

	# blur to help with edge detection
	# image = cv2.blur(image,(2,2))
	image = cv2.GaussianBlur(image,(5,5),0)

	# https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html
	edged = cv2.Canny(
		image, # input image
		50, # minVal
		200 # maxVal
	)

	# https://docs.opencv.org/3.3.1/d4/d73/tutorial_py_contours_begin.html

	# opencv 3.1
	if StrictVersion(cv2.__version__) < StrictVersion('4.0'):
		img, cnts, hierarchy = cv2.findContours(
			edged.copy(), # source image
			cv2.RETR_LIST, # contour retrieval mode
			cv2.CHAIN_APPROX_SIMPLE # contour approximation method
		)

	# opencv 4.x
	elif StrictVersion(cv2.__version__) >= StrictVersion('4.0'):
		cnts, hierarchy = cv2.findContours(
			edged.copy(), # source image
			cv2.RETR_LIST, # contour retrieval mode
			cv2.CHAIN_APPROX_SIMPLE # contour approximation method
		)

	# sort the contours
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

	#TODO: NEED TO CONFIRM CONTOUR LARGE ENOUGH

	# find rectangle?
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		if len(approx) == 4:
			screenCnt = approx
			break

	# warp
	warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

	# DEBUG and QA
	logger.debug(f'orientation_ratio: {warped.shape[1] / warped.shape[0]}, rect peri: {peri}')
	if (warped.shape[1] / warped.shape[0]) > 1 and peri < 1500:
		error_filepath = re.sub(r'(.*)/(.*)', r'\1/ERROR__rect_peri_too_small__\2', output_filepath)
		cv2.imwrite(error_filepath, orig)
		raise Exception(f'ERROR: found rect peri seems too small: {peri}')

	logger.debug(f'warped shape: {warped.shape}')
	if warped.shape[0] < 1700:
		error_filepath = re.sub(r'(.*)/(.*)', r'\1/ERROR__rect_height_too_small__\2', output_filepath)
		cv2.imwrite(error_filepath, orig)
		raise Exception(f'ERROR: found rectangle height seems too small: {warped.shape}')

	# preview
	if preview:
		imgplot = plt.imshow(warped)

	# write
	else:
		cv2.imwrite(output_filepath, warped)
		return output_filepath





