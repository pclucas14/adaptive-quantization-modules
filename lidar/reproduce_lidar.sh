
suffix='Fin'
lr='6e-4'
rth='7e-5'

CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --suffix $suffix  --n_iters 8 --recon_th $rth --dataset processed_kitti --data_size 2 40 512 ---layer_0 --num_codebooks 5 --downsample 4 --num_embeddings 256 --learning_rate $lr  &
CUDA_VISIBLE_DEVICES=1 python main.py --seed 1 --suffix $suffix  --n_iters 8 --recon_th $rth --dataset processed_kitti --data_size 2 40 512 ---layer_0 --num_codebooks 5 --downsample 4 --num_embeddings 256 --learning_rate $lr  &
CUDA_VISIBLE_DEVICES=2 python main.py --seed 2 --suffix $suffix  --n_iters 8 --recon_th $rth --dataset processed_kitti --data_size 2 40 512 ---layer_0 --num_codebooks 5 --downsample 4 --num_embeddings 256 --learning_rate $lr  &
CUDA_VISIBLE_DEVICES=3 python main.py --seed 3 --suffix $suffix  --n_iters 8 --recon_th $rth --dataset processed_kitti --data_size 2 40 512 ---layer_0 --num_codebooks 5 --downsample 4 --num_embeddings 256 --learning_rate $lr  &


CUDA_VISIBLE_DEVICES=4 python main.py --seed 0 --suffix $suffix  --n_iters 10 --recon_th $rth --dataset processed_kitti --data_size 2 40 512 ---layer_0 --num_codebooks 4 --downsample 4 --num_embeddings 256 --learning_rate $lr  &
CUDA_VISIBLE_DEVICES=5 python main.py --seed 1 --suffix $suffix  --n_iters 10 --recon_th $rth --dataset processed_kitti --data_size 2 40 512 ---layer_0 --num_codebooks 4 --downsample 4 --num_embeddings 256 --learning_rate $lr  &
CUDA_VISIBLE_DEVICES=6 python main.py --seed 2 --suffix $suffix  --n_iters 10 --recon_th $rth --dataset processed_kitti --data_size 2 40 512 ---layer_0 --num_codebooks 4 --downsample 4 --num_embeddings 256 --learning_rate $lr  &
CUDA_VISIBLE_DEVICES=7 python main.py --seed 3 --suffix $suffix  --n_iters 10 --recon_th $rth --dataset processed_kitti --data_size 2 40 512 ---layer_0 --num_codebooks 4 --downsample 4 --num_embeddings 256 --learning_rate $lr  

