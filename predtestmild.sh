PYTHONPATH="/notebooks/tensorflow/mingqiu/NTIRECompetition"
export PYTHONPATH
python3 predicttest_wavelet_advh.py  --datadir=data/DIV2K_2018/DIV2K2018_test/DIV2K_test_LR_mild \
                     --waveletimgsize=512 \
                     --hrimgsize=2048 \
                     --wavelet=db1 \
                     --scale=4 \
                     --layers=64 \
                     --featuresize=256 \
                     --reusedir=result/mild_crop100_layer64_f256_3ch_new/mild_crop100_layer64_f256_3ch_new \
                     --step=15000 \
                     --outdir=out