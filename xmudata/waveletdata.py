from xmuutil import utils
import numpy as np


class WaveletData(object):
    def __init__(self, data, wavelet_img_size=8, wavelet='db1'):
        self.data = data
        self.wavelet = wavelet
        self.wavelet_image_size = wavelet_img_size

    def get_test_set(self, batch_size):
        x_imgs, y_imgs = self.data.get_test_set(batch_size)
        if x_imgs!=None and y_imgs!=None:
            h, _, channel = np.shape(x_imgs[0])
            if h > self.wavelet_image_size :
                x_dwt_imgs = utils.get_dwt_images(x_imgs, self.wavelet_image_size, self.wavelet)

            else:
                x_dwt_imgs = x_imgs

            y_dwt_imgs = utils.get_dwt_images(y_imgs, self.wavelet_image_size, self.wavelet)
            return x_dwt_imgs, y_dwt_imgs
        else:
            return None,None

    def get_batch(self, batch_size):
        x_imgs, y_imgs = self.data.get_batch(batch_size)
        h, _, channel = np.shape(x_imgs[0])
        if h > self.wavelet_image_size:
            x_dwt_imgs = utils.get_dwt_images(x_imgs, self.wavelet_image_size, self.wavelet)
        else:
            x_dwt_imgs = x_imgs

        y_dwt_imgs = utils.get_dwt_images(y_imgs, self.wavelet_image_size, self.wavelet)
        return x_dwt_imgs, y_dwt_imgs
