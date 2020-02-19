
# Ms Pacman
python test.py --rl_env mspacman --mem_size 50000 --always_compress 1  --n_iters 5 --recon_th 0.003 --data_size 3 210 160 ---layer_0 --downsample 2 --learning_rate 5e-3


# Pong
python test.py --rl_env pong --mem_size 50000 --always_compress 1 --n_iters 5 --recon_th 0.003 --data_size 3 210 160 ---layer_0 --downsample 2 --learning_rate 1e-3

python test.py --rl_env pitfall --mem_size 50000 --always_compress 1 --n_iters 5 --recon_th 0.003 --data_size 3 210 160 ---layer_0 --downsample 2 --learning_rate 1e-3
