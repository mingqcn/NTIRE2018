PYTHONPATH="/notebooks/tensorflow/mingqiu/NTIRECompetition"
export PYTHONPATH
python3 predict_wavelet_advh.py  --groundtruth=data/DIV2K/DIV2K_valid_HR \
                     --datadir=data/DIV2K_2018/DIV2K_valid_LR_mild \
                     --postfixlen=3 \
                     --waveletimgsize=512 \
                     --hrimgsize=2048 \
                     --wavelet=db1 \
                     --scale=4 \
                     --layers=64 \
                     --featuresize=256 \
                     --reusedir=result/mild_crop100_layer64_f256_3ch_new/mild_crop100_layer64_f256_3ch_new \
                     --step=15000 \
                     --outdir=out