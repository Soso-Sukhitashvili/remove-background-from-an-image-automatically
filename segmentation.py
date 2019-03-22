import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from skimage import io
import numpy as np
import cv2

class Tree_segmentation:
    def __init__(self, image_file=None, thresh=0.2):

        if image_file==None:
            raise RuntimeError('Image location is not given! or incorrect location is provided'.upper())
        np_img = io.imread(image_file)
        hsv_img = rgb2hsv(np_img)
        hue_img = hsv_img[:, :, 0]
        binary_img = hue_img > thresh
        binary_img_3d = np.zeros(np_img.shape)
        binary_img_3d[:, :, 0] = binary_img
        binary_img_3d[:, :, 1] = binary_img
        binary_img_3d[:, :, 2] = binary_img
        mask = binary_img_3d.astype(np.uint8)
        np_img = np_img.astype(np.uint8)
        mask_bn = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        labeled_img_ch = cv2.bitwise_and(np_img, np_img, mask=mask_bn)
        self.labeled_img_ch = labeled_img_ch.astype(np.uint8)

    def save(self, dir_name=None):
        if dir==None:
            raise RuntimeError('provide location & name!'.upper())
        io.imsave(dir_name, self.labeled_img_ch)


    def __call__(self, plot=True, figsize=(8,8)):
        if plot==True:
            self.plotter(self.labeled_img_ch, figsize)
        else:
            return self.labeled_img_ch

    def plotter(self, numpy_image, figsize):

        fig, ax0 = plt.subplots(figsize=figsize)

        ax0.imshow(numpy_image)
        ax0.set_title("Labled Original Image")
        ax0.axis('off')

        fig.tight_layout()
        plt.show()

