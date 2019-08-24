
# generic
import logging
import os
from statistics import mean
import time

# modules
from crop import crop_and_rotate
from enhance import sharpen_image
from split import split_page


'''
%load_ext autoreload
%autoreload 2
%pylab
input_path = 'data/natural_light'
output_path = 'data/natural_light_done'

from bookscanner import *
process_issue_folder(input_path, output_path)
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


    # get metrics
    img_num = len(os.listdir(input_path))
    logger.debug(f'about to crop {img_num} images')

    # average time per page, estimated time remaining
    page_times = []
    errors = []

    # loop through images
    for i, page in enumerate(os.listdir(input_path)):

        # bump i
        i += 1

        # timing
        stime = time.time()

        # page path
        input_page_path = os.path.join(input_path, page)
        output_page_path = os.path.join(output_path, page)
        logging.debug('working on scanned page: %s, %s/%s' % (page,i,img_num))

        try:
            # crop and rotate
            logger.debug('cropping and rotating')
            crop_and_rotate(input_page_path, output_page_path)

            # sharpen
            logger.debug('sharpening')
            sharpen_image(output_page_path, output_page_path)

            # split
            logger.debug('splitting page')
            split_page(output_page_path)

        except:
            errors.append(input_path)
            logger.debug(f'FAILURE: {input_path}')

        # report
        page_times.append(time.time()-stime)
        logger.debug(f'tpp: {page_times[-1]}, average tpp: {mean(page_times)}, page: {i}/{img_num}, error pct: {len(errors) / i}, est total time: {int(mean(page_times) * (img_num-i))}')

    # report
    #TODO: write report to JSON to follow up
    logger.debug('\n\nfinis')
    logger.debug(f'average tpp: {mean(page_times)}, error pct: {len(errors) / img_num}')









