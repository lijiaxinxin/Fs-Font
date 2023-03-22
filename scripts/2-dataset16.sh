export CUDA_VISIBLE_DEVICES=0
python3 train.py \
    2-dataset16 \
    cfgs/0002-dataset16_custom.yaml \
    #--resume \path\to\pretrain.pdparams
