import argparse
from xmumodel.waveletsr import WaveletSR
import tensorflow as tf
import tensorlayer as tl
import os
import sys
from xmuutil import utils
import shutil
import numpy as np
import math
import tqdm

FLAGS=None

def main(_):
    if os.path.exists(FLAGS.outdir):
        shutil.rmtree(FLAGS.outdir)
    os.mkdir(FLAGS.outdir)

    img_files = sorted(os.listdir(FLAGS.datadir))
    lr_imgs= tl.visualize.read_images(img_files, FLAGS.datadir)

    network = WaveletSR(FLAGS.layers, FLAGS.featuresize, FLAGS.scale, FLAGS.waveletimgsize, FLAGS.hrimgsize, channels = 3)
    network.buildModel()
    network.resume(FLAGS.reusedir, global_step=FLAGS.step)

    for i in tqdm(range(len(img_files))):
        #output = predict(i, input_imgs, network, target_imgs, fo)
        output = ensem_predict([lr_imgs[i]], network)

        output_img = np.clip(output[0],0,1)
        output_img = output_img * 255 + 0.5
        print("%d->%s"%(i,img_files[i]))
        tl.vis.save_image(output_img, FLAGS.outdir + '/' + img_files[i])

    return


def ensem_predict(input_imgs, network):
    # ensembling
    outs_list = []
    for _, flip_axis in enumerate([0, 1, 2, -1]):
        for _, rotate_rg in enumerate([0, 90]):
            en_imgs = utils.enhance_imgs(input_imgs, rotate_rg, flip_axis)
            outs = network.predict(en_imgs)
            composed_img = utils.compose_dwt_images(outs, FLAGS.wavelet)
            anti_outs = utils.anti_enhance_imgs(composed_img, rotate_rg, flip_axis)
            outs_list.append(anti_outs[0])
    output = np.mean(outs_list, axis=0)
    return [output]


def predict(input_imgs, network):
    out0 = network.predict(input_imgs)
    output = utils.compose_dwt_images(out0, FLAGS.wavelet)
    return output


def make_same_shape(img1, img2):
    '''
    make img2 as the same shape of img1 (fill in zero)
    :param img1:
    :param img2:
    :return: padded or cropped img2
    '''
    hr_h, hr_w, _ = np.shape(img1)
    output_h, output_w, _ = np.shape(img2)
    h_pad = hr_h - output_h
    w_pad = hr_w - output_w
    output_img = img2
    if h_pad > 0:
        output_img = np.pad(output_img, pad_width=(
            (h_pad // 2, math.ceil(h_pad / 2)), (0, 0), (0, 0)),
                            mode='constant', constant_values=0)
    if w_pad > 0:
        output_img = np.pad(output_img, pad_width=(
            (0, 0), (w_pad // 2, math.ceil(w_pad / 2)), (0, 0)),
                            mode='constant', constant_values=0)
    if h_pad < 0:
        output_img = output_img[h_pad // (-2): -1 * math.ceil(h_pad / (-2)), :]
    if w_pad < 0:
        output_img = output_img[:, w_pad // (-2): -1 * math.ceil(w_pad / (-2))]
    return output_img



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--datadir", default='data/DIV2K_2018/DIV2K2018_test/DIV2K_test_LR_x8')
    parser.add_argument("--scale",default=8,type=int)
    parser.add_argument("--layers",default=32,type=int)
    parser.add_argument("--featuresize",default=256,type=int)
    parser.add_argument("--reusedir",default='result/x8_crop100_layer32_f256_3ch/x8_crop100_layer32_f256_3ch_1')
    parser.add_argument("--outdir", default='out')
    parser.add_argument("--hrimgsize",default=2048,type=int)
    parser.add_argument("--waveletimgsize",default=256,type=int)
    parser.add_argument("--wavelet", default='db1')
    parser.add_argument("--step", default=13500, type=int)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
