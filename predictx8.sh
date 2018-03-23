PYTHONPATH="/notebooks/tensorflow/mingqiu/NTIRECompetition"
export PYTHONPATH
python3 predict_wavelet_3ch.py  --groundtruth=data/DIV2K/DIV2K_valid_HR \
                     --datadir=data/DIV2K_2018/DIV2K_valid_LR_x8 \
                     --postfixlen=2 \
                     --waveletimgsize=256 \
                     --hrimgsize=2048 \
                     --wavelet=db1 \
                     --scale=8 \
                     --layers=32 \
                     --featuresize=256 \
                     --reusedir=result/x8_crop100_layer32_f256_3ch/x8_crop100_layer32_f256_3ch_1 \
                     --step=13500 \
                     --outdir=out