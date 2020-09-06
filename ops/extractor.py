import cv2
import numpy as np
import time
import utils


class PatchExtractor(object):
    @staticmethod
    def extract_positive_patches_from_tumor_region(wsi_image, tumor_gt_mask, level_used,
                                                   bounding_boxes, patch_save_dir, patch_prefix,
                                                   patch_index):
        """

            Extract positive patches targeting annotated tumor region

            Save extracted patches to desk as .png image files

            :param wsi_image:
            :param tumor_gt_mask:
            :param level_used:
            :param bounding_boxes: list of bounding boxes corresponds to tumor regions
            :param patch_save_dir: directory to save patches into
            :param patch_prefix: prefix for patch name
            :param patch_index:
            :return:
        """

        mag_factor = pow(2, level_used)
        tumor_gt_mask = cv2.cvtColor(tumor_gt_mask, cv2.COLOR_BGR2GRAY)
        print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))

        for bounding_box in bounding_boxes:
            b_x_start = int(bounding_box[0])
            b_y_start = int(bounding_box[1])
            b_x_end = int(bounding_box[0]) + int(bounding_box[2])
            b_y_end = int(bounding_box[1]) + int(bounding_box[3])
            X = np.random.random_integers(b_x_start, high=b_x_end, size=utils.NUM_POSITIVE_PATCHES_FROM_EACH_BBOX)
            Y = np.random.random_integers(b_y_start, high=b_y_end, size=utils.NUM_POSITIVE_PATCHES_FROM_EACH_BBOX)

            for x, y in zip(X, Y):
                if int(tumor_gt_mask[y, x]) != 0:
                    x_large = x * mag_factor
                    y_large = y * mag_factor
                    patch = wsi_image.read_region((x_large, y_large), 0, (utils.PATCH_SIZE, utils.PATCH_SIZE))
                    patch.save(patch_save_dir + patch_prefix + str(patch_index), 'PNG')
                    patch_index += 1
                    patch.close()

        return patch_index

    @staticmethod
    def extract_negative_patches_from_normal_wsi(wsi_image, image_open, level_used,
                                                 bounding_boxes, patch_save_dir, patch_prefix,
                                                 patch_index):
        """
            Extract negative patches from Normal WSIs

            Save extracted patches to desk as .png image files

            :param wsi_image:
            :param image_open:
            :param level_used:
            :param bounding_boxes: list of bounding boxes corresponds to detected ROIs
            :param patch_save_dir: directory to save patches into
            :param patch_prefix: prefix for patch name
            :param patch_index:
            :return:

        """

        mag_factor = pow(2, level_used)

        print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))

        for bounding_box in bounding_boxes:
            b_x_start = int(bounding_box[0])
            b_y_start = int(bounding_box[1])
            b_x_end = int(bounding_box[0]) + int(bounding_box[2])
            b_y_end = int(bounding_box[1]) + int(bounding_box[3])
            np.random.seed(int(time.time()))
            X = np.random.random_integers(b_x_start, high=b_x_end, size=utils.NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX)
            Y = np.random.random_integers(b_y_start, high=b_y_end, size=utils.NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX)

            for x, y in zip(X, Y):
                if int(image_open[y,x]):
                    x_large = x * mag_factor
                    y_large = y * mag_factor
                    patch = wsi_image.read_region((x_large, y_large), 0, (utils.PATCH_SIZE, utils.PATCH_SIZE))
                    patch.save(patch_save_dir + patch_prefix + str(patch_index), 'PNG')
                    patch_index += 1
                    patch.close()
        return patch_index

    @staticmethod
    def extract_negative_patches_from_tumor_wsi(wsi_image, tumor_gt_mask, image_open, level_used,
                                                bounding_boxes, patch_save_dir, patch_prefix,
                                                patch_index):
        """
            From Tumor WSIs extract negative patches from Normal area (reject tumor area)
            Save extracted patches to desk as .png image files

            :param wsi_image:
            :param tumor_gt_mask:
            :param image_open: morphological open image of wsi_image
            :param level_used:
            :param bounding_boxes: list of bounding boxes corresponds to tumor regions
            :param patch_save_dir: directory to save patches into
            :param patch_prefix: prefix for patch name
            :param patch_index:
            :return:

        """

        mag_factor = pow(2, level_used)
        tumor_gt_mask = cv2.cvtColor(tumor_gt_mask, cv2.COLOR_BGR2GRAY)
        print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))

        for bounding_box in bounding_boxes:
            b_x_start = int(bounding_box[0])
            b_y_start = int(bounding_box[1])
            b_x_end = int(bounding_box[0]) + int(bounding_box[2])
            b_y_end = int(bounding_box[1]) + int(bounding_box[3])
            X = np.random.random_integers(b_x_start, high=b_x_end, size=utils.NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX)
            Y = np.random.random_integers(b_y_start, high=b_y_end, size=utils.NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX)

            for x, y in zip(X, Y):
                if int(image_open[y, x]) == 1:
                    x_large = x * mag_factor
                    y_large = y * mag_factor
                    if int(tumor_gt_mask[y, x]) == 0:  # mask_gt does not contain tumor area
                        patch = wsi_image.read_region((x_large, y_large), 0, (utils.PATCH_SIZE, utils.PATCH_SIZE))
                        patch.save(patch_save_dir + patch_prefix + str(patch_index), 'PNG')
                        patch_index += 1
                        patch.close()

        return patch_index