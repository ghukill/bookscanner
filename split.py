# split

import cv2
from distutils.version import StrictVersion
import imutils
import logging
import matplotlib as plt
import numpy as np
from PIL import Imag
import os



def split_page(
	input_filepath,
	cross_section_height=10,
	kernel_size=3):

	print(input_filepath)

	# read image from filepath
	image = cv2.imread(input_filepath,0)

	# get horizontal gradient
	# sobelx = cv2.Sobel(
	# 	image[0:cross_section_height, 100:3142],
	# 	cv2.CV_64F,
	# 	1,
	# 	0,
	# 	ksize=kernel_size
	# )
	sobelx = cv2.Sobel(
		image,
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
	print(cl_index / image.shape[1])

	return (col_sums, cl_index)



