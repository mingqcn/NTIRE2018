PYTHONPATH="/notebooks/tensorflow/mingqiu/NTIRECompetition"
export PYTHONPATH
python3 trainDWT_3ch.py  --groundtruthdir=data/benchmark/augmentation/BSD100/HR \
                     --datadir=data/benchmark/augmentation/BSD100/LR \
                     --valid_datadir=data/benchmark/BSD100/LR \
                     --valid_groundtruthdir=data/benchmark/BSD100/HR \
                     --postfixlen=3 \
                     --validpostfixlen=3 \
                     --waveletimgsize=32 \
                     --imgsize=32 \
                     --wavelet=db1 \
                     --scale=4 \
                     --layers=32 \
                     --featuresize=256 \
                     --batchsize=50 \
                     --savedir=ckpt/bsd_crop32_layer32_f256_3ch \
                     --logdir=bsd_crop32_layer32_f256_3ch \
                     --iterations=400000