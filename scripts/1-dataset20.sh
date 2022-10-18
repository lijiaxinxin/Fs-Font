export CUDA_VISIBLE_DEVICES=3
python3 train.py \
    1-dataset20 \
    cfgs/1-dataset20_custom.yaml \
    #--resume \path\to\pretrain.pdparams
