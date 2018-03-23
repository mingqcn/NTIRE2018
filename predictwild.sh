PYTHONPATH="/notebooks/tensorflow/mingqiu/NTIRECompetition"
export PYTHONPATH
python3 predict_wavelet_deblur_3ch.py  --groundtruth=data/DIV2K/DIV2K_valid_HR \
                     --datadir=data/DIV2K_2018/DIV2K_valid_LR_wild \
                     --postfixlen=3 \
                     --waveletimgsize=512 \
                     --hrimgsize=2048 \
                     --wavelet=db1 \
                     --scale=4 \
                     --layers=24 \
                     --featuresize=256 \
                     --reusedir=result/wild_crop128_layer24_f256_3ch/wild_crop128_layer24_f256_3 \
                     --outdir=out