import cv2
import numpy as np
from skimage import filters
from skimage.segmentation import random_walker

def remove_noise_fun(img, border_mask=None):
    numLabels, labels, stats, _ = cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S)
    for label in range(1,numLabels):
        if stats[label,4] < 40:
            labels[labels==label] = 0
    
    height, width = labels.shape[0], labels.shape[1]


    if border_mask is not None:
        labels[border_mask==0] = 0

    return labels

def estimate_foreground_mask(depthmap, downsample_res=512, debug=False, 
                             remove_noise=False, morph_size=3, sobel_threshold=120, 
                             border_mask=None, fill_holes=True):
    
    #cv2.imwrite("debug/mask.jpg", mask)
    #Normalize
    depthmap = cv2.medianBlur(depthmap, 9)
    depthmap = (depthmap-np.min(depthmap))/(np.max(depthmap)-np.min(depthmap))*255
    
    depthmap_erosion = cv2.erode(depthmap, np.ones((morph_size,morph_size),np.uint8), iterations = 2)
    depthmap_dilation = cv2.dilate(depthmap, np.ones((morph_size,morph_size),np.uint8), iterations = 2)

    #Edge detection+threshold on foreground/background
    foreground_edge = filters.sobel(depthmap_erosion).astype(np.uint8)
    foreground_edge = cv2.threshold(foreground_edge, sobel_threshold, 255, cv2.THRESH_BINARY)[1]
    

    background_edge = filters.sobel(depthmap_dilation).astype(np.uint8)
    background_edge = cv2.threshold(background_edge, sobel_threshold, 255, cv2.THRESH_BINARY)[1]

    if remove_noise:
        background_edge = remove_noise_fun(background_edge, border_mask=border_mask)
        foreground_edge = remove_noise_fun(foreground_edge, border_mask=border_mask)
    
    markers = np.zeros(depthmap.shape, dtype=np.uint8)

    #Mark Foreground
    markers[foreground_edge>0] = 1
    markers[background_edge>0] = 2
    
    # Downsample for faster processing
    if downsample_res<=depthmap.shape[0]:
        mk = cv2.resize(markers,(downsample_res,downsample_res), interpolation = cv2.INTER_NEAREST)
        dm = cv2.resize(depthmap,(downsample_res,downsample_res), interpolation = cv2.INTER_NEAREST)
    else:
        mk = markers
        dm = depthmap

    try:
        labels = random_walker(dm, mk, beta=10, mode='bf')
    except:
        return np.zeros(depthmap.shape, dtype=np.uint8)
    
    mask = cv2.threshold(labels, 1, 255, cv2.THRESH_BINARY_INV)[1]

    if fill_holes:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        
        final_result = np.ones(mask.shape[:2])
        final_result = cv2.drawContours(final_result, contours, -1, color=(0, 255, 0), thickness=cv2.FILLED) * 255
        final_result = 255 - final_result
        mask = final_result.astype(np.uint8)
    
    return mask

if __name__ == '__main__':
    depthmap = cv2.imread('debug/depthmap_normalized_filtered.jpg', cv2.IMREAD_GRAYSCALE)
    depthmap = cv2.resize(depthmap, (512, 512))

    foreground_mask = estimate_foreground_mask(depthmap, downsample_res=255, debug=True, remove_noise=True, morph_size=5)