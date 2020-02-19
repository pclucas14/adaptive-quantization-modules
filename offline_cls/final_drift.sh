CUDA=1
seeds=( 0 1 2 3 4 )
rths=( 0.1 0.075 0.05 0.025 0.01 )

logdir='drift_appendix'

for seed in "${seeds[@]}"
do
    for rth in "${rths[@]}"
    do
        CUDA_VISIBLE_DEVICES=$CUDA python test_drift.py --seed $seed --run_dir $logdir --num_blocks 1 --recon_th $rth --mem_size 1000 --dataset miniimagenet --data_size 3 128 128 ---layer_0 --downsample 4 --num_codebooks 2 --num_embeddings 256 --learning_rate 1e-3

        CUDA_VISIBLE_DEVICES=$CUDA python test_drift.py --seed $seed --freeze_embeddings 0 --mask_unfrozen 0  --run_dir $logdir --num_blocks 1 --recon_th $rth --mem_size 1000 --dataset miniimagenet --data_size 3 128 128 ---layer_0 --downsample 4 --num_codebooks 2 --num_embeddings 256 --learning_rate 1e-3
    done
done
