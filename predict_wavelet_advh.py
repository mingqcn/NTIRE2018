import argparse
from xmumodel.waveletavdh import WaveletSR
import tensorflow as tf
import tensorlayer as tl
import os
import sys
from xmuutil import utils
import shutil
import numpy as np
import math

FLAGS=None

def main(_):
    if os.path.exists(FLAGS.outdir):
        shutil.rmtree(FLAGS.outdir)
    os.mkdir(FLAGS.outdir)

    img_files = sorted(os.listdir(FLAGS.datadir))
    lr_imgs, hr_imgs, lr_pos, hr_pos = utils.get_image_set(img_files, input_dir = FLAGS.datadir,ground_truth_dir=FLAGS.groundtruth, hr_image_size = 0, scale = FLAGS.scale, postfix_len=FLAGS.postfixlen)
    hr_norm_imgs = utils.normalize_color(hr_imgs)
    network = WaveletSR(FLAGS.layers, FLAGS.featuresize, FLAGS.scale, channels = 3)
    network.buildModel()
    network.resume(FLAGS.reusedir, global_step=FLAGS.step)

    level = FLAGS.hrimgsize // (FLAGS.scale * FLAGS.waveletimgsize)
    fo = open(FLAGS.outdir+'/psnr.csv', 'w')
    fo.writelines("seq, file, L1, PSNR\n")
    mean_list = []
    step = FLAGS.scale ** 2

    for i in range(len(img_files)):
        size, _, _ = np.shape(lr_imgs[i])
        size_hr,_,_ = np.shape(hr_imgs[i])
        target_imgs = utils.get_dwt_images([hr_norm_imgs[i]], img_size = 1 + (size_hr // (FLAGS.scale * math.pow(2,level-1))), wavelet= FLAGS.wavelet)
        input_imgs = utils.get_dwt_images([lr_imgs[i]], img_size = 1 + (size // level), wavelet= FLAGS.wavelet) if level > 1 else [lr_imgs[i]]


        #output = predict(input_imgs, network, step)
        output = ensem_predict(input_imgs,network,step)

        output_img = make_same_shape(hr_imgs[i], output[0])
        print('%dth composed image, loss = %.6f, min = %.6f, max = %.6f, mean = %.6f, var = %.6f' % (i,np.mean(np.abs(hr_imgs[i] / 255, output_img.astype(np.float32))), np.min(output_img), np.max(output_img), np.mean(output_img),math.sqrt(np.var(output_img))))

        output_img = np.clip(output_img,0,1)
        output_img = output_img * 255 + 0.5

        mean = utils.psnr_np(hr_imgs[i], output_img, scale= FLAGS.scale)


        fo.writelines("%s, %.6f\n"%(img_files[i], mean))
        #fo.writelines("%d, %s, %.6f, %.6f\n"%(i, img_files[i], np.mean(np.abs(hr_imgs[i] - output_img)), mean))
        mean_list.append(mean)
        tl.vis.save_image(output_img, FLAGS.outdir + '/' + img_files[i])

    fo.writelines("%d, Average,0, %.6f\n"%(-1, np.mean(mean_list)))
    fo.close()
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

    parser.add_argument("--datadir", default='data/DIV2K_2018/DIV2K_valid_LR_x8')
    parser.add_argument("--groundtruth",default='data/DIV2K/DIV2K_valid_HR')
    parser.add_argument("--postfixlen", default=2,type=int)
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
