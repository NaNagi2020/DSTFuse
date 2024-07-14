import os
import copy
import cv2
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np

def visualize(preds, im_path):
    if not os.path.exists(im_path):
        return
    im= resize(rgb2gray(imread(im_path)), (640, 640))
    if isinstance(preds, np.ndarray):
        for item in preds:
            pt1, pt2 = int(item[0]), int(item[1])
            pt3, pt4 = int(item[2]), int(item[3])
            cv2.rectangle(im, (pt1, pt2), (pt3, pt4), (0,0,255), 2)
        plt.imshow(im)
        plt.show()
    else:
        for pred in preds:
            if pred['boxes'].device == 'cpu':
                boxes = pred['boxes'].numpy()
            else:
                boxes = pred['boxes'].cpu().detach().numpy()
            for item in boxes:
                pt1, pt2 = int(item[0]), int(item[1])
                pt3, pt4 = int(item[2]), int(item[3])
                cv2.rectangle(im, (pt1, pt2), (pt3, pt4), (0,0,255), 2)
            plt.imshow(im)
            plt.show()
