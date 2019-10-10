# 1 block HP
rths=( 0.021 0.019 0.023 0.025 0.017 0.015 )
for rth in "${rths[@]}"
do
    CUDA_VISIBLE_DEVICES=0 python offline_main.py --eps_th 0.005  --run_dir 'HPsearch_last' --override_cl_defaults --n_classes_per_task 5 --multiple_heads 0 --num_blocks 1 --recon_th $rth --mem_size 1000 --dataset miniimagenet --n_classes 100 --data_size 3 128 128 ---layer_0 --commitment_cost 2 --decay 0.6 --downsample 4 --embed_grad_update 0 --learning_rate 1e-3 --num_codebooks 2 --num_embeddings 128 --quant_size 1 1 --stride 2
    CUDA_VISIBLE_DEVICES=1 python offline_main.py --eps_th 0.005  --run_dir 'HPsearch_last' --override_cl_defaults --n_classes_per_task 5 --multiple_heads 0 --num_blocks 1 --recon_th $rth --mem_size 1000 --dataset miniimagenet --n_classes 100 --data_size 3 128 128 ---layer_0 --commitment_cost 2 --decay 0.6 --downsample 4 --embed_grad_update 0 --learning_rate 1e-3 --num_codebooks 1 --num_embeddings 128 --quant_size 1  --stride 2
done



# 2 bloch HP
rthsa=( 0.025 0.0225 0.02 0.02   0.02   0.0175 )
rthsb=( 0.02  0.02   0.02 0.0175 0.015  0.015  )


for i in $(seq 0 5);
do
    CUDA_VISIBLE_DEVICES=0 python offline_main.py --eps_th 0.005 --run_dir 'HPsearch_last' --override_cl_defaults --n_classes_per_task 5 --multiple_heads 0 --num_blocks 2 --recon_th ${rthsa[$i]} ${rthsb[$i]} --mem_size 1000 --dataset miniimagenet --n_classes 100 --data_size 3 128 128 ---layer_0 --commitment_cost 2 --decay 0.6 --embed_grad_update 0 --learning_rate 1e-3 --num_embeddings 128 --quant_size 1 1 --num_codebooks 2 --stride 2 --downsample 4 ---layer_1 --commitment_cost 2 --decay 0.6 --downsample 1 --embed_grad_update 0 --learning_rate 1e-3 --num_codebooks 1 --num_embeddings 128 --quant_size 1 --stride 1
done 
