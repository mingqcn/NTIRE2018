from xmudata.DIV2K2018 import DIV2K2018
from xmudata.normdata import NormalizeData
import argparse
from xmumodel.waveletsr import WaveletSR
import tensorflow as tf
import sys
from xmudata.waveletdata import WaveletData

FLAGS=None

def main(_):
    data = DIV2K2018(train_data_dir= FLAGS.datadir, image_size=FLAGS.imgsize,train_truth_dir=FLAGS.groundtruthdir,
                            test_data_dir=FLAGS.valid_datadir, test_truth_dir=FLAGS.valid_groundtruthdir,scale=FLAGS.scale,
                            train_postfix_len=FLAGS.postfixlen, test_postfix_len=FLAGS.validpostfixlen, test_per=0.1)
    data = NormalizeData(data, x_normal = (FLAGS.imgsize != FLAGS.waveletimgsize))
    dwt_data =WaveletData(data, wavelet_img_size=FLAGS.waveletimgsize, wavelet =FLAGS.wavelet)
    
    network = WaveletSR(FLAGS.layers, FLAGS.featuresize, FLAGS.scale, FLAGS.waveletimgsize, FLAGS.imgsize * FLAGS.scale,channels= 3 )
    network.buildModel()
    network.set_data(dwt_data)
    
    reuse = False if FLAGS.reusedir==None else True
    network.train(FLAGS.batchsize, FLAGS.iterations,save_dir=FLAGS.savedir, log_dir=FLAGS.log ,reuse=reuse, reuse_dir=FLAGS.reusedir, reuse_epoch=FLAGS.reuseep)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--groundtruthdir",default="data/DIV2K_2018/augmentation/DIV2K_train_HR")
    """
    datadir                               postfix_len scale  track
    data/DIV2K_2018/DIV2K_train_LR_x8          2        8    1: bicubic downscaling x8 competition
    data/DIV2K_2018/DIV2K_train_LR_mild        3        4    2: realistic downscaling x4 with mild conditions competition
    data/DIV2K_2018/DIV2K_train_LR_difficult   3        4    3: realistic downscaling x4 with difficult conditions competition
    data/DIV2K_2018/DIV2K_train_LR_wild        4/       4    4: wild downscaling x4 competition
    """
    parser.add_argument("--datadir",default="data/DIV2K_2018/augmentation/DIV2K_train_LR_x8")
    parser.add_argument("--valid_datadir", default='data/DIV2K_2018/DIV2K_valid_LR_x8')
    parser.add_argument("--valid_groundtruthdir", default='data/DIV2K/DIV2K_valid_HR')
    parser.add_argument("--postfixlen",default=2, type=int)
    parser.add_argument("--validpostfixlen",default=2, type=int)
    parser.add_argument("--waveletimgsize",default=100,type=int)
    parser.add_argument("--wavelet",default='db1')
    parser.add_argument("--imgsize",default=100,type=int)
    parser.add_argument("--scale",default=8,type=int)
    parser.add_argument("--layers",default=32,type=int)
    parser.add_argument("--featuresize",default=128,type=int)
    parser.add_argument("--batchsize", default=2, type=int)
    parser.add_argument("--savedir",default='ckpt/x8_crop100_layer32_f128_3ch')
    parser.add_argument("--reusedir",default=None)
    parser.add_argument("--reuseep", default=0, type = int)
    parser.add_argument("--iterations",default=5000,type=int)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)