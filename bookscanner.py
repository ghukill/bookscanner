
# generic
import logging
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


# setup logger
logging.basicConfig(level=logging.DEBUG)
# parso shims
logging.getLogger('parso.python.diff').disabled = True
logging.getLogger('parso.cache').disabled = True
logger = logging.getLogger(__name__)



def process_issue_folder(input_path, output_path):

	'''

	1. crop out black background
	2. sharpen images

	'''


	# handle input/output folders
	if not os.path.exists(input_path) or not os.path.isdir(input_path):
		raise Exception('%s either does not exist, or is not a directory' % input_path)
	if not os.path.exists(output_path):
		os.makedirs(output_path)


	# loop through images
	for page in os.listdir(input_path):

		# page path
		page_path = os.path.join(input_path, page)
		logging.debug('working on page: %s' % page_path)



