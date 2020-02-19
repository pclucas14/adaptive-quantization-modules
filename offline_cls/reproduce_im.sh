seeds=( 0 1 2 3 4 )

""" 1 Block """

# This is for the final classifier performance results. Averaged over 5 runs
for seed in "${seeds[@]}"
do
    CUDA_VISIBLE_DEVICES=$CUDA python offline_main.py --run_dir 'final_1b' --seed $seed --num_blocks 1 --recon_th .05 --mem_size 1000 --dataset miniimagenet --data_size 3 128 128 ---layer_0 --downsample 4 --num_codebooks 1 --num_embeddings 256 --learning_rate 1e-3
done

""" 2 Block """
# This is for the final classifier performance results. Averaged over 5 runs
for seed in "${seeds[@]}"
do
    CUDA_VISIBLE_DEVICES=$CUDA python offline_main.py --seed $seed --run_dir 'final_2b' --num_blocks 2 --recon_th .1 .03 --mem_size 1000 --dataset miniimagenet --data_size 3 128 128 ---layer_0 --downsample 4 --num_codebooks 1 --num_embeddings 256 --learning_rate 1e-3 ---layer_1 --downsample 1 --num_codebooks 1 --num_embeddings 32 --learning_rate 1e-3
done

