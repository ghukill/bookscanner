# enhance

from PIL import Image, ImageFilter


def sharpen_image(input_filepath, output_filepath):

	print('sharpening...')

	# open image
	img = Image.open(input_filepath)

	# sharpen
	img_sharp = img.filter(ImageFilter.SHARPEN)

	# write
	img_sharp.save(output_filepath)