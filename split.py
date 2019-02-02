# split

import cv2
from distutils.version import StrictVersion
import imutils
import logging
import matplotlib as plt
import numpy as np
from PIL import Image
import os


# setup logger
logging.basicConfig(level=logging.DEBUG)
# parso shims
logging.getLogger('parso.python.diff').disabled = True
logging.getLogger('parso.cache').disabled = True
logger = logging.getLogger(__name__)


def split_page(
	input_filepath,
	cross_section_height=20,
	cross_section_origin='bottom',
	kernel_size=3,
	mid_range=None,
	split_image=True,
	split_padding=10,
	delete_original=True):

	'''
	Focus on area roughly in the middle, 1500-2500
		- will be better to handle % of image.shape
	'''


	# read image from filepath
	image = cv2.imread(input_filepath, 0)

	# make sure double page
	if image.shape[1] / image.shape[0] > 1.25:

		# debug
		metrics = {
			'filepath':input_filepath,
			'shape':image.shape,
		}

		# if cross section None, use full height
		if cross_section_height == None:
			cross_section_height = image.shape[0]

		# set mid-section interval
		if mid_range == None:
			logger.debug('determining mid-range pixels')
			mid_range = [int(image.shape[1]/2 - 200), int(image.shape[1]/2 + 200)]
			metrics['mid_range'] = mid_range

		# get section to work with
		if cross_section_origin == 'top':
			image_cross_section = image[0:cross_section_height, mid_range[0]:mid_range[1]]
		elif cross_section_origin == 'bottom':
			image_cross_section = image[-cross_section_height:image.shape[0], mid_range[0]:mid_range[1]]

		# get horizontal gradient
		sobelx = cv2.Sobel(
			image_cross_section,
			cv2.CV_64F,
			1,
			0,
			ksize=kernel_size
		)

		# sum columns
		col_sums = np.sum(sobelx, axis=0)

		# plot
		# plt.pyplot.plot(col_sums)

		# get center line
		cl_index_rel = np.argmax(col_sums)
		cl_index = mid_range[0] + cl_index_rel
		metrics['cl_index'] = cl_index

		# check
		metrics['halfway_percentage'] = cl_index / image.shape[1]

		# split image
		if split_image:

			# get filename root
			filename_root, filename_ext = input_filepath.split('/')[-1].split('.')

			# split left page
			l_img = image[0:image.shape[0], 0:(cl_index + split_padding)]
			l_filepath = input_filepath.replace(filename_root, '%s_0' % filename_root)
			cv2.imwrite(l_filepath, l_img)

			# split right page
			r_img = image[0:image.shape[0], (cl_index - split_padding):image.shape[1]]
			r_filepath = input_filepath.replace(filename_root, '%s_1' % filename_root)
			cv2.imwrite(r_filepath, r_img)

			# delete original
			if delete_original:
				os.remove(input_filepath)

		# debug
		logger.debug(metrics)
		return metrics

	else:
		logger.debug('appears to be single page')
		return {
			'filepath':input_filepath,
			'shape':image.shape,
		}



