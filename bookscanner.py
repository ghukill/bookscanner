
# generic
import os

# modules
from crop import *
from enhance import *


'''
%load_ext autoreload
%autoreload 2
%pylab
input_path = 'data/natural_light'
output_path = 'data/natural_light_done'
'''



def process_issue_folder(input_path, output_path):

	'''

	1. crop out black background
	2. sharpen images

	'''

	# confirm folder
	if not os.path.exists(input_path) or not os.path.isdir(input_path):
		raise Exception('%s either does not exist, or is not a directory' % input_path)


	# create output folder
	if not os.path.exists(output_path):
		os.makedirs(output_path)


	# crop



