python vq_main.py --run_dir final_cifar --n_runs 20 --data_size 3 32 32 --n_classes 10 --recon_th 1. --rehearsal 1 --mem_size 200 --n_iters 1 --num_epochs 1 --optimization blockwise --num_blocks 1 --dataset split_cifar10 ---layer_0 --learning_rate 5e-3  --downsample 2 --stride 2 --quant_size 1 --commitment_cost 6 --num_embeddings 128 --num_codebooks 1 --embed_grad_update 1

python vq_main.py --run_dir final_cifar --n_runs 20 --data_size 3 32 32 --n_classes 10 --recon_th 1. --rehearsal 1 --mem_size 500 --n_iters 1 --num_epochs 1 --optimization blockwise --num_blocks 1 --dataset split_cifar10 ---layer_0 --learning_rate 5e-3 --downsample 2 --stride 2 --quant_size 1 --commitment_cost 4. --num_embeddings 128 --num_codebooks 1 --embed_grad_update 1

# Riemer et al. baseline
python vq_main.py --run_dir final_cifar --n_runs 20 --data_size 3 32 32 --n_classes 10 --recon_th 1000 --rehearsal 1 --mem_size 200 --n_iters 1 --num_epochs 1 --optimization blockwise --num_blocks 1 --dataset split_cifar10 ---layer_0 --learning_rate 5e-3  --downsample 2 --stride 2 --num_embeddings 256 --num_codebooks 1 --model gumbel

python vq_main.py --run_dir final_cifar --n_runs 20 --data_size 3 32 32 --n_classes 10 --recon_th 1000 --rehearsal 1 --mem_size 500 --n_iters 1 --num_epochs 1 --optimization blockwise --num_blocks 1 --dataset split_cifar10 ---layer_0 --learning_rate 5e-3  --downsample 2 --stride 2 --num_embeddings 32 --num_codebooks 1 --model gumbel
