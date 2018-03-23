from xmuutil import utils


class NormalizeData(object):
    ''''
    normalize data in preposing process
    '''
    def __init__(self, data, x_normal = True, y_normal = True):
        '''
        init function
        :param data: data
        :param x_normal: True for x normalization
        :param y_normal: True for y normalization
        '''
        self.data = data
        self.x_normal = x_normal
        self.y_normal = y_normal

    def get_batch(self, batch_size):
        x_imgs, y_imgs = self.data.get_batch(batch_size)
        x_norm_imgs = utils.normalize_color(x_imgs) if self.x_normal else x_imgs
        y_norm_imgs = utils.normalize_color(y_imgs) if self.y_normal else y_imgs
        return x_norm_imgs, y_norm_imgs

    def get_test_set(self, batch_size):
        x_imgs, y_imgs = self.data.get_test_set(batch_size)
        x_norm_imgs = utils.normalize_color(x_imgs) if self.x_normal else x_imgs
        y_norm_imgs = utils.normalize_color(y_imgs) if self.y_normal else y_imgs
        return x_norm_imgs,y_norm_imgs


