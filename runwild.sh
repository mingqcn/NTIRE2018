PYTHONPATH="/notebooks/tensorflow/mingqiu/NTIRECompetition"
export PYTHONPATH
python3 trainDWT_avdh.py  --groundtruthdir=data/DIV2K_2018/augmentation/DIV2K_train_HR \
                     --datadir=data/DIV2K_2018/augmentation/DIV2K_train_LR_wild \
                     --valid_datadir=data/DIV2K_2018/DIV2K_valid_LR_wild \
                     --valid_groundtruthdir=data/DIV2K/DIV2K_valid_HR \
                     --postfixlen=4 \
                     --validpostfixlen=3 \
                     --waveletimgsize=100 \
                     --imgsize=100 \
                     --wavelet=db1 \
                     --scale=4 \
                     --layers=64 \
                     --featuresize=256 \
                     --batchsize=1 \
                     --savedir=ckpt/wild_crop100_layer64_f256_3ch_new \
                     --logdir=wild_crop100_layer64_f256_3ch_new \
                     --iterations=400000