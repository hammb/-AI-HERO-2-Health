import numpy as np

def post_process(segmentation, threshold=10):
    segmentation = segmentation.astype(int)
    sizes = np.bincount(segmentation.flat) # sizes of 
    too_small = sizes<threshold
    mask = too_small[segmentation]
    segmentation[mask] = 0
    return segmentation