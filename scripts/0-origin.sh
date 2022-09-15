#export CUDA_VISIBLE_DEVICES=4
python3 train.py \
    0-origin \
    cfgs/my_custom.yaml \
    #--resume \path\to\pretrain.pdparams
