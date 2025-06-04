gpu=5
# Patching for t-PatchGNN
python run_density_prep.py \
    --dataset physionet --batch_size 32 --n 12000 \
    --alpha 2.0 --history 24 --pred_window 24 --gpu $gpu \
    --patch_ts --patch_size 8 --stride 8 \

# No patching for others
python run_density_prep.py \
    --dataset physionet --batch_size 32 --n 12000 \
    --alpha 2.0 --history 24 --pred_window 24 --gpu $gpu \
    --patch_size 8 --stride 8 \

python run_density_prep.py \
    --dataset mimic --batch_size 32 --n 23457 \
    --alpha 2.0 --history 24 --pred_window 24 --gpu $gpu \
    --patch_ts --patch_size 8 --stride 8 \

python run_density_prep.py \
    --dataset mimic --batch_size 32 --n 23457 \
    --alpha 2.0 --history 24 --pred_window 24 --gpu $gpu \
    --patch_size 8 --stride 8 \

python run_density_prep.py \
    --dataset activity --batch_size 32 --n 25 \
    --alpha 2.0 --history 3000 --pred_window 1000 --gpu $gpu \
    --patch_ts --patch_size 300 --stride 300 \

python run_density_prep.py \
    --dataset activity --batch_size 32 --n 25 \
    --alpha 2.0 --history 3000 --pred_window 1000 --gpu $gpu \
    --patch_size 300 --stride 300 \

python run_density_prep.py \
    --dataset ushcn --batch_size 192 --n 1114 \
    --alpha 2.0 --history 24 --pred_window 1 --gpu $gpu \
    --patch_ts --patch_size 2 --stride 2 \

python run_density_prep.py \
    --dataset ushcn --batch_size 192 --n 1114 \
    --alpha 2.0 --history 24 --pred_window 1 --gpu $gpu \
    --patch_size 2 --stride 2 \
