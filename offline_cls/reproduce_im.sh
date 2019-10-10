# 1 Block

rth=0.025

seeds=( 0 1 2 3 4 5 6 7 8 9 )
for seed in "${seeds[@]}"
do
    CUDA_VISIBLE_DEVICES=0 python offline_main.py --seed $seed --eps_th 0.005  --run_dir 'Final' --override_cl_defaults --n_classes_per_task 5 --multiple_heads 0 --num_blocks 1 --recon_th $rth --mem_size 1000 --dataset miniimagenet --n_classes 100 --data_size 3 128 128 ---layer_0 --commitment_cost 2 --decay 0.6 --downsample 4 --embed_grad_update 0 --learning_rate 1e-3 --num_codebooks 1 --num_embeddings 128 --quant_size 1  --stride 2
done


# 2 Block 

rthsa=0.025
rthsb=0.02

seeds=( 0 1 2 3 4 5 6 7 8 9 )
for seed in "${seeds[@]}"
do
    CUDA_VISIBLE_DEVICES=1 python offline_main.py --seed $seed --eps_th 0.005 --run_dir 'Final' --override_cl_defaults --n_classes_per_task 5 --multiple_heads 0 --num_blocks 2 --recon_th ${rthsa[$i]} ${rthsb[$i]} --mem_size 1000 --dataset miniimagenet --n_classes 100 --data_size 3 128 128 ---layer_0 --commitment_cost 2 --decay 0.6 --embed_grad_update 0 --learning_rate 1e-3 --num_embeddings 128 --quant_size 1 1 --num_codebooks 2 --stride 2 --downsample 4 ---layer_1 --commitment_cost 2 --decay 0.6 --downsample 1 --embed_grad_update 0 --learning_rate 1e-3 --num_codebooks 1 --num_embeddings 128 --quant_size 1 --stride 1
done 
