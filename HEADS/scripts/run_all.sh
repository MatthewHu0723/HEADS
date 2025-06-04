patience=10
gpu=2
startseed=1

for model in tPatchGNN GraFITi mTAN
do
    python run.py \
    --plugin --model $model \
    --lamb 0.5 --alpha 2.0 \
    --dataset physionet --n 12000 --history 24 \
    --patience $patience --batch_size 32 --lr 1e-3 \
    --learn-emb \
    --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 64 \
    --outlayer Linear --seed $startseed --gpu $gpu


    python run.py \
    --plugin --model $model\
    --lamb 0.8 --alpha 2.0 \
    --dataset mimic --n 23457 --history 24 \
    --patience $patience --batch_size 32 --lr 1e-3 \
    --learn-emb \
    --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 64 \
    --outlayer Linear --seed $startseed --gpu $gpu


    python run.py \
    --plugin --model $model \
    --lamb 0.1 --alpha 2.0 \
    --dataset activity --n 25 --history 3000 \
    --patience $patience --batch_size 32 --lr 1e-3 \
    --learn-emb \
    --patch_size 300 --stride 300 --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 32 \
    --outlayer Linear --seed $startseed --gpu $gpu 

    python run.py \
    --plugin --model $model \
    --lamb 0.7 --alpha 2.0 \
    --dataset ushcn --n 1114 --history 24 \
    --patience $patience --batch_size 192 --lr 1e-3 \
    --learn-emb \
    --patch_size 2 --stride 2 --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 32 \
    --outlayer Linear --seed $startseed --gpu $gpu

done
