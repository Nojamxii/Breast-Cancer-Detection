import sys
import glob
import os.path
import threading
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import utils as utils
from ops.WSIOps import WSIOps 
import shutil
import matplotlib.pyplot as plt


class Queue(object):
    def __init__(self):
        self.patches = []
        self.pos = []

    def put(self, patch, pos):
        self.patches.append(patch)
        self.pos.append(pos)
        assert self.patches.__len__() == self.pos.__len__(), 'Both queue lengths must be same.'

    def get_next_batch(self):
        assert self.patches.__len__() == self.pos.__len__(), 'Both queue lengths must be same.'
        cnt = utils.BATCH_SIZE if self.patches.__len__() > utils.BATCH_SIZE else self.patches.__len__()
        items = self.patches[:cnt]
        del self.patches[:cnt]
        del self.pos[:cnt]
        return items

    def clear(self):
        self.patches = []
        self.pos = []

    def print(self):
        print(self.pos)


def extract_patch_from_bb(thread_index, bounding_box, wsi_image, image_open, level_used, heat_map_dir):
    """

     mapping from (x,y) -> (raw, col)
     x -> col
     y -> row

    """
    # factor to map low res cords into high res
    mag_factor = pow(2, level_used)
    b_x_start = bounding_box[0]
    b_y_start = bounding_box[1]
    b_x_end = bounding_box[0] + bounding_box[2]
    b_y_end = bounding_box[1] + bounding_box[3]
    col_cords = np.arange(b_x_start, b_x_end)
    row_cords = np.arange(b_y_start, b_y_end)
    print('Apx. patch count for thread(%d): %d' % (thread_index, len(row_cords) * len(col_cords)))

    for row in row_cords:
        for col in col_cords:
            if int(image_open[row, col]) is not utils.PIXEL_BLACK:
                wsi_patch = wsi_image.read_region((col * mag_factor, row * mag_factor), 0,
                                                  (utils.PATCH_SIZE, utils.PATCH_SIZE))
                file_name = str(row) + '_' + str(col) + '_' + str(level_used)
                wsi_patch.save(os.path.join(heat_map_dir, file_name) +'.jpg', 'PNG')
                wsi_patch.close()

def extract_patches(wsi_image_path, wsi_image_name, wsi_mask_path=None):
    print('extract_patches(): %s' % wsi_image_name)

    heatmap_patch_dir = os.path.join(utils.HEAT_MAP_RAW_PATCHES_DIR, wsi_image_name)
    if not os.path.exists(heatmap_patch_dir):
        os.makedirs(heatmap_patch_dir)
    else:
        shutil.rmtree(heatmap_patch_dir)
        os.mkdir(heatmap_patch_dir)


    if wsi_mask_path is None:
        wsi_image, rgb_image, level_used = wsi_ops.read_wsi_normal(wsi_image_path)
        assert wsi_image is not None, 'Failed to read Whole Slide Image %s.' % wsi_image_name
    else:
        wsi_image, rgb_image, _, _, level_used = wsi_ops.read_wsi_tumor(wsi_image_path, wsi_mask_path)
        assert wsi_image is not None, 'Failed to read Whole Slide Image %s.' % wsi_image_name


    bounding_boxes, rgb_contour, image_open = wsi_ops.find_roi_bbox(np.array(rgb_image))

    plt.imsave(utils.HEAT_MAP_WSIs_PATH + '/' + wsi_image_name + 'lowest_level.png',rgb_image)
    plt.imsave(utils.HEAT_MAP_WSIs_PATH + '/' + wsi_image_name + 'contour_img.png',rgb_contour)
    
    

    print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))

    coord = tf.train.Coordinator()

    threads = []
    for thread_index in range(len(bounding_boxes)):
    # for thread_index in range(0,1):
        args = (thread_index, bounding_boxes[thread_index], wsi_image, image_open, level_used, heatmap_patch_dir)
        t = threading.Thread(target=extract_patch_from_bb, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    wsi_image.close()
    sys.stdout.flush()


def extract_patches_tumor():
    wsi_image_names = glob.glob(os.path.join(utils.TRAIN_TUMOR_WSI_PATH, '*.tif'))
    wsi_image_names.sort()
    wsi_mask_names = glob.glob(os.path.join(utils.TRAIN_TUMOR_MASK_PATH, '*.tif'))
    wsi_mask_names.sort()

    image_mask_pair = zip(wsi_image_names, wsi_mask_names)
    image_mask_pair = list(image_mask_pair)
    for image_path, mask_path in image_mask_pair:
        extract_patches(image_path, utils.get_filename_from_path(image_path), mask_path)


def extract_patches_normal():
    wsi_image_names = glob.glob(os.path.join(utils.TRAIN_NORMAL_WSI_PATH, '*.tif'))
    wsi_image_names.sort()

    for image_path in wsi_image_names:
        extract_patches(image_path, utils.get_filename_from_path(image_path))


def extract_patches_test():
    wsi_image_names = glob.glob(utils.TEST_WSI_DIR + '/*.tif')
    wsi_image_names.sort()
    print(wsi_image_names)
    # wsi_image_names = wsi_image_names[1:2]
    for image_path in wsi_image_names:
        extract_patches(image_path, utils.get_filename_from_path(image_path))
    print('Finished extract patches.')


if __name__ == '__main__':
    wsi_ops = WSIOps()
    # extract_patches_tumor()
    # extract_patches_normal()
    '''
    extract patches from WSI image for test
    '''
    extract_patches_test()
