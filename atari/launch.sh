
# what I used for Ms Pacman
python test.py --mem_size 1e9 --always_compress 1  --n_iters 5 --recon_th 0.003 --mem_size 1000 --data_size 3 210 160 ---layer_0 --downsample 2 --learning_rate 5e-3


# pong (same but lower lr). Works for Pitfall as well
python test.py --mem_size 1e9 --always_compress 1 --n_iters 5 --recon_th 0.003 --mem_size 1000 --data_size 3 210 160 ---layer_0 --downsample 2 --learning_rate 1e-3
