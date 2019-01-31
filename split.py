# split

import cv2
from distutils.version import StrictVersion
import imutils
import logging
import matplotlib as plt
import numpy as np
from PIL import Image
import os



def split_page(
	input_filepath,
	cross_section_height=500,
	kernel_size=3):

	'''
	Focus on area roughly in the middle, 1500-2500
		- will be better to handle % of image.shape
	'''


	# read image from filepath
	image = cv2.imread(input_filepath,0)

	# make sure double page
	if image.shape[1] - image.shape[0] > 1000:

		# debug
		print(input_filepath, image.shape)

		if cross_section_height == None:
			cross_section_height = image.shape[0]

		# get horizontal gradient
		sobelx = cv2.Sobel(
			image[0:cross_section_height, 1500:2500],
			cv2.CV_64F,
			1,
			0,
			ksize=kernel_size
		)

		# sum columns
		col_sums = np.sum(sobelx, axis=0)

		# plot
		# plt.plot(col_sums)

		# get center line
		cl_index = np.argmax(col_sums)

		# check
		print( (cl_index + 1500) / image.shape[1])

		return (col_sums, cl_index)



