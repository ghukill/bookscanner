# enhance

import logging
from PIL import Image, ImageFilter


# setup logger
logging.basicConfig(level=logging.DEBUG)
# parso shims
logging.getLogger('parso.python.diff').disabled = True
logging.getLogger('parso.cache').disabled = True
logger = logging.getLogger(__name__)


def sharpen_image(input_filepath, output_filepath):

	# open image
	img = Image.open(input_filepath)

	# sharpen
	img_sharp = img.filter(ImageFilter.SHARPEN)

	# write
	img_sharp.save(output_filepath)

	# return
	return output_filepath