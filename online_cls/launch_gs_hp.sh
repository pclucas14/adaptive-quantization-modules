n_embeds=(2 4 8 16 32 64 128 256 512 )
lrs=( 1e-3 3e-3 5e-3 )


for n_embed in "${n_embeds[@]}"
do
    for lr in "${lrs[@]}"
    do
    python vq_main.py --run_dir hp_gs --n_runs 20 --data_size 3 32 32 --n_classes 10 --recon_th 1000 --rehearsal 1 --mem_size 200 --n_iters 1 --num_epochs 1 --optimization blockwise --num_blocks 1 --dataset split_cifar10 ---layer_0 --learning_rate $lr  --downsample 2 --stride 2 --num_embeddings $n_embed --num_codebooks 1 --model gumbel

    python vq_main.py --run_dir hp_gs --n_runs 20 --data_size 3 32 32 --n_classes 10 --recon_th 1000 --rehearsal 1 --mem_size 500 --n_iters 1 --num_epochs 1 --optimization blockwise --num_blocks 1 --dataset split_cifar10 ---layer_0 --learning_rate $lr  --downsample 2 --stride 2 --num_embeddings $n_embed --num_codebooks 1 --model gumbel
    done
done

