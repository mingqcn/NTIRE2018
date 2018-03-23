PYTHONPATH="/notebooks/tensorflow/mingqiu/NTIRECompetition"
export PYTHONPATH
python3 predict_wavelet_3ch.py  --groundtruthdir=data/benchmark/BSD100/HR \
                     --datadir=data/benchmark/BSD100/LR \
                     --postfixlen=3 \
                     --waveletimgsize=512 \
                     --hrimgsize=2048 \
                     --wavelet=db1 \
                     --scale=4 \
                     --layers=32 \
                     --featuresize=256 \
                     --reusedir=result/bsd_crop32_layer32_f256_3ch/bsd_crop32_layer32_f256_3ch \
                     --outdir=out