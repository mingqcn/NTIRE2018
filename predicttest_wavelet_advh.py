import argparse
from xmumodel.waveletavdh import WaveletSR
import tensorflow as tf
import tensorlayer as tl
import os
import sys
from xmuutil import utils
import shutil
import numpy as np
from tqdm import tqdm

FLAGS=None

def main(_):
    if os.path.exists(FLAGS.outdir):
        shutil.rmtree(FLAGS.outdir)
    os.mkdir(FLAGS.outdir)

    img_files = sorted(os.listdir(FLAGS.datadir))
    lr_imgs= tl.visualize.read_images(img_files, FLAGS.datadir)
    network = WaveletSR(FLAGS.layers, FLAGS.featuresize, FLAGS.scale, channels = 3)
    network.buildModel()
    network.resume(FLAGS.reusedir, global_step=FLAGS.step)

    step = FLAGS.scale ** 2

    for i in tqdm(range(len(img_files))):
        #output = predict(input_imgs, network, step)
        output = ensem_predict([lr_imgs[i]],network,step)

        output_img = np.clip(output[0],0,1)
        output_img = output_img * 255 + 0.5
        print("%d->%s"%(i,img_files[i]))
        tl.vis.save_image(output_img, FLAGS.outdir + '/' + img_files[i])

    return


def concat(out_hfreq, out_lowfreq, step):
    low = out_lowfreq[0]
    high = out_hfreq[0]
    out0 = np.concatenate([low[:, :, 0:1], high[:, :, 0:step - 1]], 2)
    out1 = np.concatenate([low[:, :, 1:2], high[:, :, step - 1: 2 * (step - 1)]], 2)
    out2 = np.concatenate([low[:, :, 2:3], high[:, :, 2 * step - 2: 3 * (step - 1)]], 2)
    out = np.concatenate([out0, out1, out2], 2)
    return out

def ensem_predict(input_imgs, network, step):
    # ensembling
    outs_list = []
    for _, flip_axis in enumerate([0, 1, 2, -1]):
        for _, rotate_rg in enumerate([0, 90]):
            en_imgs = utils.enhance_imgs(input_imgs, rotate_rg, flip_axis)
            out_hfreq, out_lowfreq = network.predict(en_imgs)
            out = concat(out_hfreq, out_lowfreq, step)
            composed_img = utils.compose_dwt_images([out], FLAGS.wavelet)
            anti_outs = utils.anti_enhance_imgs(composed_img, rotate_rg, flip_axis)
            outs_list.append(anti_outs[0])
    output = np.mean(outs_list, axis=0)
    return [output]


def predict(input_imgs, network, step):
    out_hfreq, out_lowfreq = network.predict(input_imgs)
    out = concat(out_hfreq, out_lowfreq, step)
    output = utils.compose_dwt_images([out], FLAGS.wavelet)
    return output




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--datadir", default='data/DIV2K_2018/DIV2K2018_test/DIV2K_test_LR_x8')
    parser.add_argument("--scale",default=8,type=int)
    parser.add_argument("--layers",default=32,type=int)
    parser.add_argument("--featuresize",default=256,type=int)
    parser.add_argument("--reusedir",default='ckpt/x8_crop100_layer32_f256_3ch')
    parser.add_argument("--outdir", default='out')
    parser.add_argument("--hrimgsize",default=2048,type=int)
    parser.add_argument("--waveletimgsize",default=256,type=int)
    parser.add_argument("--wavelet", default='db1')
    parser.add_argument("--step", default=10000, type=int)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
