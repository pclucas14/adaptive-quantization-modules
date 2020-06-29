quality=(1 5 10 20 40 50 60 70 80 90 95)
sizes=(500) # 200 500)
CUDA=1

for q in "${quality[@]}"
do
    for mem_size in "${sizes[@]}"
    do
        echo 'hyperparameter search loop'
        # CUDA_VISIBLE_DEVICES=$CUDA python vq_main.py --run_dir jpeg_baseline --jpeg_quality $q --n_runs 10  --sunk_cost  --jpeg_baseline 1  --data_size 3 32 32 --n_classes 10 --recon_th -1 --rehearsal 1 --mem_size $mem_size --num_blocks 1 --dataset split_cifar10 ---layer_0 --downsample 2 --num_embeddings 8 
    done
done
        
echo 'running best runs for 20 runs'
CUDA_VISIBLE_DEVICES=0 python vq_main.py --run_dir jpeg_baseline_final --jpeg_quality 10 --n_runs 20  --sunk_cost  --jpeg_baseline 1  --data_size 3 32 32 --n_classes 10 --recon_th -1 --rehearsal 1 --mem_size 200 --num_blocks 1 --dataset split_cifar10 ---layer_0 --downsample 2 --num_embeddings 8 &

CUDA_VISIBLE_DEVICES=1 python vq_main.py --run_dir jpeg_baseline_final --jpeg_quality 80 --n_runs 20  --sunk_cost  --jpeg_baseline 1  --data_size 3 32 32 --n_classes 10 --recon_th -1 --rehearsal 1 --mem_size 500 --num_blocks 1 --dataset split_cifar10 ---layer_0 --downsample 2 --num_embeddings 8 
