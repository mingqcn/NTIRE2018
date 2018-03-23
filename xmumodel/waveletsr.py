from xmumodel.model import Model
import tensorflow as tf
import tensorlayer.layers as tl
from xmuutil.scalelayer import ScaleLayer
from xmuutil.relulayer import ReluLayer
from xmuutil.normalizelayer import NormalizeLayer
from xmuutil import utils

"""
An implementation of WaveletSR used for
super-resolution of images as described in:

`WaveletSR: Image Super-Resolution Using Wavelet Decomposition Tree`
(http://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf)

"""


class WaveletSR(Model):

    def __init__(self, num_layers, feature_size, scale, wavelet_img_size, hr_img_size, channels = 3 ):
        lr_img_size = hr_img_size // scale
        input_channels = channels * (lr_img_size // wavelet_img_size) ** 2 if lr_img_size > wavelet_img_size else channels
        output_channels =  channels * (hr_img_size // wavelet_img_size) ** 2
        Model.__init__(self, num_layers, feature_size, scale, input_channels, output_channels)

    def buildModel(self):
        print("Building WaveletSR Model...")

        # input layers
        x = tl.InputLayer(self.input, name='inputlayer')

        # One convolution before res blocks and to convert to required feature depth
        x = tl.Conv2d(x, self.feature_size, [3, 3], name='1st_conv')

        # Store the output of the first convolution to add later
#        conv_1 = x

        """
        This creates `num_layers` number of resBlocks

        """


        scaling_factor = 0.1

        # Add the residual blocks to the model
        for i in range(self.num_layers):
            x = self.__resBlock(x, self.feature_size, scale=scaling_factor, layer=i)

        # One more convolution, and then we add the output of our first conv layer
        x = tl.Conv2d(x, self.feature_size, [3, 3], act=None, name='m1')
 #       x = tl.ElementwiseLayer([conv_1, x], tf.add, name='res_add')

        # One final convolution
        # One convolution after res blocks
        output = tl.Conv2d(x, self.output_channels, [1, 1], act=None, name='lastLayer')
        self.output = output.outputs

        self.cacuLoss(output)

        # Tensorflow graph setup... session, saver, etc.
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        print("Done building! ")


    def cacuLoss(self, x):
        '''
        caculate the loss, and write it to tensorboard;
        :param x: output tensor
        :return: None
        '''
        loss = tf.losses.absolute_difference(self.target, x.outputs)
        #Using adam optimizer as mentioned in the paper
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        #This is the train operation for our objective
        self.train_op = optimizer.minimize(loss)

        PSNR = utils.psnr_tf(self.target, x.outputs)

        streaming_loss, self.streaming_loss_update = tf.contrib.metrics.streaming_mean(loss)
        streaming_loss_scalar = tf.summary.scalar('loss',streaming_loss)

        streaming_psnr, self.streaming_psnr_update = tf.contrib.metrics.streaming_mean(PSNR)
        streaming_psnr_scalar = tf.summary.scalar('PSNR',streaming_psnr)
        self.test_merge = tf.summary.merge([streaming_loss_scalar])


        # Scalar to keep track for loss
        summary_loss = tf.summary.scalar("loss_in_all", loss)
        summaries = [summary_loss]
        _, _, _, channel = self.target.get_shape()
        for i in range(channel // 3):
            loss =tf.losses.absolute_difference(self.target[:,:,:,i] , x.outputs[:,:,:,i])
            summaries.append(tf.summary.scalar("loss%d" %(i), loss))
        summary_psnr = tf.summary.scalar("PSNR", PSNR)
        summaries.append(utils.variable_summeries(x.outputs,'output'))
        summaries.append(utils.variable_summeries(self.target,'target'))


        # Image summaries for input, target, and output
        #input_image = tf.summary.image("input_image", tf.cast(self.input[:,:,0:1], tf.uint8))
        #target_image = tf.summary.image("target_image", tf.cast(self.target[:,:,0:1], tf.uint8))
        #output_image = tf.summary.image("output_image", tf.cast(x.outputs[:,:,0:1], tf.uint8))

        self.train_merge = tf.summary.merge(summaries)

    def __resBlock(self, x, channels=64, kernel_size=[3, 3], scale=1, layer=0):
        """
           Creates a convolutional residual block
           as defined in the paper. More on
           this inside model.py

           a resBlock is defined in the paper as
           (excuse the ugly ASCII graph)
               x
               |\
               | \
               |  conv2d
               |  relu
               |  conv2d
               | /
               |/
               + (addition here)
               |
               result

           x: input to pass through the residual block
           channels: number of channels to compute
           stride: convolution stride

           :param x: input tensor
           :param channnels: channnels in the block
           :param kernel_size: filter kernel size
           :param scale: scale for residual skip
           :param layer: layer number
           """
        nn = tl.Conv2d(x, channels, kernel_size, act=tf.nn.relu, name='res%d/c1' % (layer))
        nn = tl.Conv2d(nn, channels, kernel_size, act=None, name='res%d/c2' % (layer))
        nn = ScaleLayer(nn, scale, name='res%d/scale' % (layer))
        n = tl.ElementwiseLayer([x, nn], tf.add, name='res%d/res_add' % (layer))
        return n

    def __resBlock_IN(self, x, channels=64, kernel_size=[3, 3], scale=1, layer=0):
        """
           Creates a convolutional residual block
           as defined in the paper. More on
           this inside model.py

           a resBlock is defined in the paper as
           (excuse the ugly ASCII graph)
               x
               |\
               | \
               |  conv2d
               |  relu
               |  IN
               |  conv2d
               | /
               |/
               + (addition here)
               |
               result

           x: input to pass through the residual block
           channels: number of channels to compute
           stride: convolution stride

           :param x: input tensor
           :param channnels: channnels in the block
           :param kernel_size: filter kernel size
           :param scale: scale for residual skip
           :param layer: layer number
           """
        nn = tl.Conv2d(x, channels, kernel_size, act=tf.nn.relu, name='res%d/c1' % (layer))
        nn = tl.Conv2d(nn, channels, kernel_size, act=None, name='res%d/c2' % (layer))
        nn = ScaleLayer(nn, scale, name='res%d/scale' % (layer))
        n = tl.ElementwiseLayer([x, nn], tf.add, name='res%d/res_add' % (layer))
        return n

    def __resBlock1(self, x, channels=64, kernel_size=[3, 3], scale=1, layer=0):
        """
           Creates a convolutional residual block
           as defined in the paper. More on
           this inside model.py

           a resBlock is defined in the paper as
           (excuse the ugly ASCII graph)
               x
               |\
               | \
               |  conv2d
               |  relu
               |  conv2d
               |  relu
               | /
               |/
               + (addition here)
               |
               result

           x: input to pass through the residual block
           channels: number of channels to compute
           stride: convolution stride

           :param x: input tensor
           :param channnels: channnels in the block
           :param kernel_size: filter kernel size
           :param scale: scale for residual skip
           :param layer: layer number
           """
        nn = tl.Conv2d(x, channels, kernel_size, act=tf.nn.relu, name='res%d/c1' % (layer))
        nn = tl.Conv2d(nn, channels, kernel_size, act=tf.nn.relu, name='res%d/c2' % (layer))
        nn = ScaleLayer(nn, scale, name='res%d/scale' % (layer))
        n = tl.ElementwiseLayer([x, nn], tf.add, name='res%d/res_add' % (layer))
        return n

    def __resBlock2(self, x, channels=64, kernel_size=[3, 3], scale=1, layer=0):
        """
           Creates a convolutional residual block
           as defined in the paper. More on
           this inside model.py

           a resBlock is defined in the paper as
           (excuse the ugly ASCII graph)
               x
               |\
               | \
               |  relu
               |  conv2d
               |  relu
               |  conv2d
               | /
               |/
               + (addition here)
               |
               result

           x: input to pass through the residual block
           channels: number of channels to compute
           stride: convolution stride

           :param x: input tensor
           :param channnels: channnels in the block
           :param kernel_size: filter kernel size
           :param scale: scale for residual skip
           :param layer: layer number
           """

        nn = ReluLayer(x, name='relu%d' % (layer))
        nn = tl.Conv2d(nn, channels, kernel_size, act=tf.nn.relu, name='res%d/c1' % (layer))
        nn = tl.Conv2d(nn, channels, kernel_size, act=None, name='res%d/c2' % (layer))
        nn = ScaleLayer(nn, scale, name='res%d/scale' % (layer))
        n = tl.ElementwiseLayer([x, nn], tf.add, name='res%d/res_add' % (layer))
        return n