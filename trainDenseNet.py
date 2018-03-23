from xmudata.DIV2K2018 import DIV2K2018
import argparse
from xmumodel.densenet import DenseNet
import tensorflow as tf
import sys

FLAGS=None

def main(_):
    data = DIV2K2018(FLAGS.datadir, FLAGS.groundtruthdir,FLAGS.imgsize,FLAGS.scale,FLAGS.postfixlen)
    network = DenseNet(FLAGS.imgsize,FLAGS.denseblock,FLAGS.growthrate,FLAGS.bottlenecksize,FLAGS.layers,FLAGS.featuresize,FLAGS.scale)
    network.buildModel()
    network.set_data(data)
    network.train(FLAGS.batchsize, FLAGS.iterations,FLAGS.savedir,False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--groundtruthdir",default="data/DIV2K_2018/DIV2K_train_HR")
    """
    datadir                               postfix_len scale  track
    data/DIV2K_2018/DIV2K_train_LR_x8          2        8    1: bicubic downscaling x8 competition
    data/DIV2K_2018/DIV2K_train_LR_mild        3        4    2: realistic downscaling x4 with mild conditions competition
    data/DIV2K_2018/DIV2K_train_LR_difficult   3        4    3: realistic downscaling x4 with difficult conditions competition
    data/DIV2K_2018/DIV2K_train_LR_wild        4        4    4: wild downscaling x4 competition
    """
    parser.add_argument("--datadir",default="data/DIV2K_2018/DIV2K_train_LR_x8")
    parser.add_argument("--postfixlen",default=2)
    parser.add_argument("--imgsize",default=32,type=int)
    parser.add_argument("--scale",default=8,type=int)
    parser.add_argument("--layers",default=8,type=int)
    parser.add_argument("--featuresize",default=16,type=int)
    parser.add_argument("--denseblock",default=4,type=int)
    parser.add_argument("--growthrate",default=16,type=int)
    parser.add_argument("--bottlenecksize",default=256,type=int)
    parser.add_argument("--batchsize",default=10,type=int)
    parser.add_argument("--savedir",default='ckpt/saved_EDSR_k8z96n1000_r')
    parser.add_argument("--reusedir",default='ckpt/saved_EDSR_k8z96n1000')
    parser.add_argument("--iterations",default=1000,type=int)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
