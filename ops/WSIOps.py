import cv2
import numpy as np
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
import matplotlib.pyplot as plt


class WSIOps(object):
    """
        # ================================
        # Class to annotate WSIs with ROIs
        # ================================

    """

    def_level = 7

    @staticmethod
    def read_wsi_mask(mask_path, level=def_level):
        try:
            wsi_mask = OpenSlide(mask_path)

            mask_image = np.array(wsi_mask.read_region((0, 0), level,
                                                       wsi_mask.level_dimensions[level]))

        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return None, None

        return wsi_mask, mask_image

    @staticmethod
    def read_wsi_normal(wsi_path):
        """
            # =====================================================================================
            # read WSI image and resize
            # Due to memory constraint, we use down sampled (4th level, 1/32 resolution) image
            # ======================================================================================
        """
        try:
            wsi_image = OpenSlide(wsi_path)
            level_used = wsi_image.level_count - 1 # 低分辨率采样
            rgb_image = np.array(wsi_image.read_region((0, 0), level_used,
                                                       wsi_image.level_dimensions[level_used]))

        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return None, None, None

        return wsi_image, rgb_image, level_used

    @staticmethod
    def read_wsi_tumor(wsi_path, mask_path):
        """
            # =====================================================================================
            # read WSI image and resize
            # Due to memory constraint, we use down sampled (4th level, 1/32 resolution) image
            # ======================================================================================
        """
        try:
            wsi_image = OpenSlide(wsi_path)
            wsi_mask = OpenSlide(mask_path)

            level_used = wsi_image.level_count - 1

            rgb_image = np.array(wsi_image.read_region((0, 0), level_used,
                                                       wsi_image.level_dimensions[level_used]))
            plt.imshow(rgb_image)

            mask_level = wsi_mask.level_count - 1
            tumor_gt_mask = wsi_mask.read_region((0, 0), mask_level,
                                                 wsi_mask.level_dimensions[mask_level])

            resize_factor = float(1.0 / pow(2, level_used - mask_level))
            print('mask resize_factor: %f' % resize_factor)
            tumor_gt_mask = cv2.resize(np.array(tumor_gt_mask), (0, 0), fx=resize_factor, fy=resize_factor)
            plt.imshow(tumor_gt_mask)
            wsi_mask.close()
        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return None, None, None, None

        return wsi_image, rgb_image, wsi_mask, tumor_gt_mask, level_used

    def find_roi_bbox_tumor_gt_mask(self, mask_image):
        mask = cv2.cvtColor(mask_image, cv2.COLOR_RGBA2GRAY)  #BGR2GRAY MODIFIED BY DZF
        plt.imshow(mask,cmap=plt.cm.gray)
        close_kernel = np.ones((20,20),dtype=np.uint8)
        mask_close = cv2.morphologyEx(np.array(mask),cv2.MORPH_CLOSE,close_kernel)
        plt.imshow(mask_close)
        # open_kernel = np.ones((5,5),dtype=np.uint8)
        # mask_open = cv2.morphologyEx(np.array(mask_close),cv2.MORPH_OPEN,open_kernel)
        bounding_boxes, _ = self.get_bbox(mask,mask_image)
        return bounding_boxes

    def find_roi_bbox(self, rgb_image):
        # hsv -> 3 channel
        BGR = cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2BGR)
        hsv = cv2.cvtColor(BGR, cv2.COLOR_BGR2HSV) #Convert BGR32 to HSV 
        lower_red = np.array([20, 20, 20])
        upper_red = np.array([200, 200, 200])
        # mask -> 1 channel
        #Make a mask to filter don't interest region
        mask = cv2.inRange(hsv, lower_red, upper_red)
        plt.imshow(mask,cmap=plt.cm.gray)
        
        #Close operation,先进行膨胀运算再进行腐蚀，填充图像的内部孔洞和图像的凹角点，把两个邻近的目标连接起来
        close_kernel = np.ones((20, 20), dtype=np.uint8)
        image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
        plt.imshow(image_close,cmap=plt.cm.gray)
        #Open operation, 先进行腐蚀运算再进行膨胀运算，消除离散点和"毛刺"
        open_kernel = np.ones((5, 5), dtype=np.uint8)
        image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
        plt.imshow(image_open,cmap=plt.cm.gray)
        bounding_boxes, rgb_contour = self.get_bbox(cont_img=image_open,rgb_image=rgb_image)
        return bounding_boxes, rgb_contour, image_open

    @staticmethod
    def get_image_open(wsi_path):
        try:
            wsi_image = OpenSlide(wsi_path)
            level_used = wsi_image.level_count - 1
            rgb_image = np.array(wsi_image.read_region((0, 0), level_used,wsi_image.level_dimensions[level_used]))
            wsi_image.close()
        except OpenSlideUnsupportedFormatError:
            raise ValueError('Exception: OpenSlideUnsupportedFormatError for %s' % wsi_path)
        '''
        with OpenSlide(wsi_path) as wsi_image:
            rgb_image = wsi_image.read_region((0,0),8,(512,512))
        '''
        # hsv -> 3 channel
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([20, 20, 20])
        upper_red = np.array([200, 200, 200])
        # mask -> 1 channel
        mask = cv2.inRange(hsv, lower_red, upper_red)

        close_kernel = np.ones((20, 20), dtype=np.uint8)
        image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
        open_kernel = np.ones((5, 5), dtype=np.uint8)
        image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
        return image_open

    @staticmethod
    def get_bbox(cont_img,rgb_image):
        '''
        Find contours using a binary image,retrieves only the extreme outer contours,
        compresses horizontal, vertical, and diagonal segments and leaves only their end points
        '''
        _, contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rgb_contour = None
        if rgb_image.shape:
            rgb_contour = rgb_image.copy()
            cv2.drawContours(rgb_contour, contours, -1, (255,255,255), 2)
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        return bounding_boxes, rgb_contour

            


    @staticmethod
    def draw_bbox(image, bounding_boxes):
        rgb_bbox = image.copy()
        for i, bounding_box in enumerate(bounding_boxes):
            x = int(bounding_box[0])
            y = int(bounding_box[1])
            cv2.rectangle(rgb_bbox, (x, y), (x + bounding_box[2], y + bounding_box[3]), color=(0, 0, 255),
                          thickness=2)
        return rgb_bbox

    @staticmethod
    def split_bbox(image, bounding_boxes, image_open):
        rgb_bbox_split = image.copy()
        for bounding_box in bounding_boxes:
            for x in range(bounding_box[0], bounding_box[0] + bounding_box[2]):
                for y in range(bounding_box[1], bounding_box[1] + bounding_box[3]):
                    if int(image_open[y, x]) == 1:
                        cv2.rectangle(rgb_bbox_split, (x, y), (x, y),
                                      color=(255, 0, 0), thickness=2)

        return rgb_bbox_split
