from xmumodel.model import Model
import tensorflow as tf
import tensorlayer.layers as tl
from xmuutil.scalelayer import ScaleLayer
from tqdm import tqdm
import os
import shutil
import xmuutil.utils as utils
"""
An implementation of WaveletSR used for
super-resolution of images as described in:

`WaveletSR: Image Super-Resolution Using Wavelet Decomposition Tree`
(http://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf)

"""


class WaveletSR(Model):

    def __init__(self, num_layers, feature_size, scale, channels = 3):
        self.num_layers = num_layers
        self.feature_size = feature_size
        self.scale = scale
        self.output_channels = channels * (scale) ** 2
        self.input_channels = channels

        #Placeholder for image inputs
        self.input = tf.placeholder(tf.float32, [None, None, None, self.input_channels], name='input')
        #Placeholder for upscaled image ground-truth
        self.target_hfreq = tf.placeholder(tf.float32, [None, None, None, self.output_channels - self.input_channels], name='output_high')
        self.target_lowfreq = tf.placeholder(tf.float32, [None, None, None, self.input_channels], name='output_low')

        self.train_high_op = None
        self.train_low_op = None
        self.train_merge = None
        self.test_merge = None

    def buildModel(self):
        print("Building WaveletSR Model...")

        # input layers
        x = tl.InputLayer(self.input, name='inputlayer')

        # One convolution before res blocks and to convert to required feature depth
        x = tl.Conv2d(x, self.feature_size, [3, 3], name='1st_conv')

        # Store the output of the first convolution to add later
        conv_1 = x

        """
        This creates half  `num_layers` number of resBlocks

        """
        scaling_factor = 0.1

        # Add the residual blocks to the model
        for i in range(self.num_layers // 2):
            x = self.__resBlock(x, self.feature_size, scale=scaling_factor, layer=i)

        # One more convolution, and then we add the output of our vhd
        hfreq = tl.Conv2d(x, self.feature_size, [3, 3], act=None, name='m1')
        output_hfreq = tl.Conv2d(hfreq, self.output_channels-self.input_channels , [1, 1], act=None, name='hfreqLayer')

        x = tl.ElementwiseLayer([conv_1, x], tf.add, name='res_add1')

        for i in range(self.num_layers // 2, self.num_layers):
            x = self.__resBlock(x, self.feature_size, scale=scaling_factor, layer=i)

        # One more convolution, and then we add the output of our a
        x = tl.Conv2d(x, self.feature_size, [3, 3], act=None, name='m2')
        x = tl.ElementwiseLayer([conv_1, x], tf.add, name='res_add2')

        # One final convolution
        # One convolution after res blocks
        output_lowfreq = tl.Conv2d(x, self.input_channels, [1, 1], act=None, name='lowfreqLayer')

        self.output_lowfreq = output_lowfreq.outputs
        self.output_hfreq = output_hfreq.outputs
        self.cacuLoss()

        # Tensorflow graph setup... session, saver, etc.
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        print("Done building! ")


    def cacuLoss(self):
        '''
        caculate the loss, and write it to tensorboard;
        :param x: output tensor
        :return: None
        '''
        loss_hfreq = tf.losses.absolute_difference(self.target_hfreq, self.output_hfreq)
        loss_lowfreq = tf.losses.absolute_difference(self.target_lowfreq, self.output_lowfreq)
        #Using adam optimizer as mentioned in the paper
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)


        #This is the train operation for our objective
        self.train_low_op = optimizer.minimize(loss_lowfreq)
        self.train_high_op = optimizer.minimize(loss_hfreq)

        streaming_loss_high, self.streaming_loss_high_update = tf.contrib.metrics.streaming_mean(loss_hfreq)
        streaming_loss_high_scalar = tf.summary.scalar('loss_high',streaming_loss_high)
        streaming_loss_low, self.streaming_loss_low_update = tf.contrib.metrics.streaming_mean(loss_lowfreq)
        streaming_loss_low_scalar = tf.summary.scalar('loss_low',streaming_loss_low)

        self.test_merge = tf.summary.merge([streaming_loss_low_scalar, streaming_loss_high_scalar])


        # Scalar to keep track for loss
        summaries = []
        with tf.name_scope("low_freq"):
            _, _, _, channel = self.target_lowfreq.get_shape()
            for i in range(channel):
                loss =tf.losses.absolute_difference(self.target_lowfreq[:,:,:,i] ,self.output_lowfreq[:,:,:,i])
                summaries.append(tf.summary.scalar("loss%d" %(i), loss))

        with tf.name_scope("high_freq"):
            _, _, _, channel = self.target_hfreq.get_shape()
            for i in range(channel):
                loss =tf.losses.absolute_difference(self.target_hfreq[:,:,:,i] ,self.output_hfreq[:,:,:,i])
                summaries.append(tf.summary.scalar("loss%d" %(i), loss))

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



    """
    Train the neural network
    """
    def train(self,batch_size= 10, iterations=1000,save_dir="saved_models",reuse=False,reuse_dir=None,reuse_epoch=None,log_dir="log"):

        #create the save directory if not exist
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.mkdir(log_dir)
        #Make new save directory
        os.mkdir(save_dir)
        #Operation to initialize all variables
        init = tf.global_variables_initializer()
        print("Begin training...")
        with self.sess as sess:
            #Initialize all variables
            sess.run(init)
            if reuse:
                self.resume(reuse_dir,reuse_epoch)
            #create summary writer for train
            train_writer = tf.summary.FileWriter(log_dir+"/train",sess.graph)

            #If we're using a test set, include another summary writer for that
            test_writer = tf.summary.FileWriter(log_dir+"/test",sess.graph)
            test_feed = []
            step = self.scale ** 2

            while True:
                test_x,test_y = self.data.get_test_set(batch_size)

                if test_x!=None and test_y!=None:
                    test_y_low, test_y_high = utils.seperate_high_low_freq(test_y, step, self.input_channels)
                    test_feed.append({
                            self.input:test_x,
                            self.target_hfreq:test_y_high,
                            self.target_lowfreq:test_y_low
                    })
                else:
                    break

            #This is our training loop
            for i in tqdm(range(iterations)):
                #Use the data function we were passed to get a batch every iteration
                x,y = self.data.get_batch(batch_size)
                y_low, y_high = utils.seperate_high_low_freq(y,step,self.input_channels)

                #Create feed dictionary for the batch
                feed = {
                    self.input:x,
                    self.target_hfreq: y_high,
                    self.target_lowfreq: y_low
                }
                #Run the train op and calculate the train summary
                summary, _, _ = sess.run([self.train_merge, self.train_high_op, self.train_low_op],feed)
                #Write train summary for this step
                train_writer.add_summary(summary,i)
                #test every 10 iterations
                if i%200 == 0:
                    sess.run(tf.local_variables_initializer())
                    for j in range(len(test_feed)):
                        sess.run([self.streaming_loss_high_update, self.streaming_loss_low_update],feed_dict=test_feed[j])
                    streaming_summ = sess.run(self.test_merge)
                    #Write test summary
                    test_writer.add_summary(streaming_summ,i)

                # Save our trained model
                if i!=0 and i % 500 == 0:
                    self.save(save_dir,i)

            self.save(save_dir)
            test_writer.close()
            train_writer.close()

    def predict(self, x):
        return self.sess.run([self.output_hfreq, self.output_lowfreq], feed_dict={self.input: x})