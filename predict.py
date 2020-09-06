from postprocess.wsi_classification import RFClassifier
from postprocess.extract_feature_heatmap import extract_features_test
import utils
import os
import shutil
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from openslide import OpenSlide
import pandas as pd
from skimage.measure import label
from skimage.measure import regionprops
import json

from postprocess.extract_feature_heatmap import extract_features

def generate_image_open(img_rgb):   
        bgr = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
        lower = np.array([20,20,20])
        upper = np.array([200,200,200])
        mask = cv2.inRange(hsv,lower,upper)
        close_kernel = np.ones((20,20),dtype=np.uint8)
        image_close = cv2.morphologyEx(np.array(mask),cv2.MORPH_CLOSE,close_kernel)
        open_kernel = np.ones((5,5),dtype=np.uint8)
        image_open = cv2.morphologyEx(np.array(image_close),cv2.MORPH_OPEN,open_kernel)

        return image_open, mask

def generate_image_open_from_grayscale(img_grayscale):
    bgr = cv2.cvtColor(img_grayscale,cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    lower = np.array([20,20,20])
    upper = np.array([200,200,200])
    mask = cv2.inRange(hsv,lower,upper)
    close_kernel = np.ones((20,20),dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(mask),cv2.MORPH_CLOSE,close_kernel)
    open_kernel = np.ones((5,5),dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close),cv2.MORPH_OPEN,open_kernel)

    return image_open, mask

def get_region_prop(heatmap,dir_name):
    rgb = np.array(heatmap.copy())
    t90 = np.array(heatmap.copy())
    t90[t90 < int(255*0.9)] = 0
    t90[t90 > int(255*0.9)] = 255

    heatmap_2d = np.array(rgb[:,:,:1]).reshape(rgb.shape[0],rgb.shape[1])
    heatmap_t90_2d = np.array(t90[:,:,:1]).reshape(t90.shape[0],t90.shape[1])
    img_labeled = label(heatmap_2d)
    # plt.imshow(heatmap_2d)
    # plt.imshow(heatmap_t90_2d)
    # plt.imshow(img_labeled)
    region_props = regionprops(img_labeled,heatmap_2d)
    
    # plt.imsave(dir_name + '/heatmap_open.png',heatmap_2d)
    # plt.imsave(dir_name + '/heatmap_t90_open.png',heatmap_t90_2d)
    

    region_coors = []
    regions = []
    for region in region_props:
        region_t90 = {}
        region_t90['area'] = region['area']
        region_t90['perimeter'] = region['perimeter']
        region_t90['eccentricity'] = region['eccentricity']
        region_t90['extent'] = region['extent']
        region_t90['solidity'] = region['solidity']
        region_t90['average_probs'] = region['mean_intensity'] / 255.0
        bbox = region['bbox']
        img_cont = heatmap_t90_2d[bbox[0]:bbox[2],bbox[2]:bbox[3]].copy()
        _, region_t90['contour_coors'], _ = cv2.findContours(img_cont,cv2.RETR_EXTERNAL, 
                                            cv2.CHAIN_APPROX_SIMPLE) 
        region_coors.append(region_t90['contour_coors'])
        regions.append(region_t90)

        return regions,region_coors
        
def draw_contour(rgb_img,line_color,):
    image_open, _ = generate_image_open(rgb_img)
    plt.imshow(image_open)
    _, contours, _ = cv2.findContours(image_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rgb_contour = rgb_img.copy()
    cv2.drawContours(rgb_contour, contours, -1, line_color, 2)
    return rgb_contour


if __name__ == '__main__':
    '''
    Input: prob_img,wsi image
    Calculate: features,feature_importance,prediction,prediction_probability,
    Return: image with tumor region contours
    '''

    prob_paths = glob.glob(utils.TEST_PROB_DIR + '/*')
    wsi_paths = glob.glob(utils.TEST_WSI_DIR + '/*')
    assert len(prob_paths) == len(wsi_paths), 'WSI images must match prob images!'
    wsi_paths.sort()
    prob_paths.sort()


    
    for prob_path, wsi_path in zip(prob_paths,wsi_paths):
        dir_name = wsi_path.split('/')[-1].split('.')[0]
        dir_name = utils.TEST_RESULT + '/' +dir_name
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.mkdir(dir_name)

        wsi_img = OpenSlide(wsi_path)
        dispaly_level = int(input('Level from 0 to %d you want to dispaly:' %(wsi_img.level_count-1)))
        rgb_img = np.array(wsi_img.read_region((0,0),dispaly_level,wsi_img.level_dimensions[dispaly_level]))[:,:,:3].copy()
        wsi_img.close()
        plt.imsave(dir_name + '/rgb.png',rgb_img)

        image_open, mask = generate_image_open(rgb_img.copy())
        plt.imsave(dir_name + '/open.png',image_open,cmap=plt.cm.gray)
        plt.imsave(dir_name + '/mask.png',mask,cmap=plt.cm.gray)
        
        prob_img = plt.imread(prob_path)
        plt.imshow(prob_img,cmap=plt.cm.gray)
        plt.imsave(dir_name + '/prob_img.png',prob_img,cmap=plt.cm.gray)
        
        # features, region_coors = extract_features(heatmap,image_open)
        # img_contour = image_open.copy()
        # cv2.drawContours(img_contour,region_coors,-1,(255,0,0),2)
        # plt.imshow(img_contour)
        # plt.imsave(dir_name + '/contour.jpg',img_contour)

        # img_contour_origin = draw_contour(rgb_img,(255,0,0))
        # img_contour_heatmap = draw_contour(heatmap,(255,255,255))
        # plt.imsave(dir_name + '/contour_origin.jpg',img_contour_origin)
        # plt.imsave(dir_name + '/contour_heatmap.jpg',img_contour_heatmap)

        features, t50, t90, coor_x, coor_y = extract_features(prob_img,image_open)
        plt.imsave(dir_name + '/t50_prob.png',t50,cmap=plt.cm.gray)
        plt.imsave(dir_name + '/t90_prob.png',t90,cmap=plt.cm.gray)

        coordinate = pd.DataFrame(columns=['x','y'])
        index = 0
        for x,y in zip(coor_x,coor_y):
            coordinate.loc[index] = [x,y]
            index += 1
        coordinate.to_csv(dir_name + '/coors.csv')
        df_features = pd.DataFrame(columns=
                                            ['region_count', 'ratio_tumor_tissue', 'largest_tumor_area', 'longest_axis_largest_tumor',
                                            'pixels_gt_90', 'avg_prediction', 'max_area', 'mean_area', 'area_variance', 'area_skew',
                                            'area_kurt', 'max_perimeter', 'mean_perimeter', 'perimeter_variance', 'perimeter_skew',
                                            'perimeter_kurt', 'max_eccentricity', 'mean_eccentricity', 'eccentricity_variance',
                                            'eccentricity_skew', 'eccentricity_kurt', 'max_extent', 'mean_extent', 'extent_variance',
                                            'extent_skew', 'extent_kurt', 'max_solidity', 'mean_solidity', 'solidity_variance',
                                            'solidity_skew', 'solidity_kurt'])
        df_features.loc[0] = features
        rf_classifier = RFClassifier(utils.RF_MODEL_PARAMETERS)
        pred,pred_proba,feature_importance = rf_classifier.predict(df_features)

        df_result = pd.DataFrame(columns=
                                          ['region_count', 'ratio_tumor_tissue', 'largest_tumor_area', 'longest_axis_largest_tumor',
                                            'pixels_gt_90', 'avg_prediction', 'max_area', 'mean_area', 'area_variance', 'area_skew',
                                            'area_kurt', 'max_perimeter', 'mean_perimeter', 'perimeter_variance', 'perimeter_skew',
                                            'perimeter_kurt', 'max_eccentricity', 'mean_eccentricity', 'eccentricity_variance',
                                            'eccentricity_skew', 'eccentricity_kurt', 'max_extent', 'mean_extent', 'extent_variance',
                                            'extent_skew', 'extent_kurt', 'max_solidity', 'mean_solidity', 'solidity_variance',
                                            'solidity_skew', 'solidity_kurt', 'pred', 'pred_prob']   )
        
        result = features
        result.append(pred)
        result.append(pred_proba)
        df_result.loc[0] = result
        df_result.to_csv(dir_name + '/result.csv')

       




        


