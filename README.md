# Online Continual Compression via Stacked Quantization Modules (SQM)
Stacking Quantization blocks for efficient lifelong online compression </br>
Code for reproducing all results in our paper, which can be found [here](paper.pdf) </br>

## (key) Requirements 
- Python 3.7
- Pytorch 1.1.0
- TensorboardX
- Mayavi (for displaying LiDARs only)

## Structure

    ├── Common 
        ├── modular.py          # Module (QLayer) and Stacked modules (QStack). Includes most key ops, such as adaptive buffer        
        ├── quantize.py         # Discretization Ops (GumbelSoftmax, Vector/Tensor Quantization and Argmax Quantization)
        ├── model.py            # Encoder, Decoder, Classifier Blocks       
    ├── Final Results           
        ├── ....                # log / results files for experiments reported in the paper 
    ├── Lidar
        ├── ....                # files to run LiDAR experiments 
    ├── Offline Cls
        ├── ....                # files to run the offline classification (e.g. Imagenet) experiments 
    ├── Online Cls              
        ├── ....                # files to run the online classification (e.g. CIFAR) experiments 
    ├── Utils             
        ├── args.py             # Contains command-line args. *ALL model configuration* is handled in this file.
        ├── buffer.py           # Basic buffer implementation. Handled raw and compressed representations
        ├── kitti_loader.py     # DataLoader for the Kitti LiDAR data. TODO: add download script for KITTI
        ├── kitti_utils.py      # Preprocessing code for LiDAR
        ├── utils.py            # Logging / Saving & Loading Models, Args
       

## Arguments usage
 - specific block parameters are separated by `---layer_i` flag
 - specific block parameters are specified AFTER regular args. 
 e.g. for a 2 block architecture
 ```
python offline_main.py --recon_th 0.008 0.015 --mem_size 1000 --dataset miniimagenet --n_classes 100 --data_size 3 128 128 ---layer_0 --commitment_cost 2 --decay 0.6 --embed_grad_update 0 --learning_rate 5e-3 --num_embeddings 128 --quant_size 1 --stride 2 --downsample 2 ---layer_1 --commitment_cost 2 --decay 0.6 --downsample 2 --embed_grad_update 0 --learning_rate 1e-3 --num_codebooks 2 --num_embeddings 128 --quant_size 1 1 --stride 2
 ```
 It is expected that `--num_blocks <value>` matches the amount of `--layer_i` flags
 
 
## Running Experiments and Logging
When running code, buffer samples and reconstructions are by default dumped in `samples` and `lidars` directory. You can run `mkdir samples; mkdir lidars` in the home directory to avoid errors before the first run. Tips on how to visualize data (especially for LiDAR) are available in the experiment specific directories. 

## Acknowledgements 
We would like to thank authors of the following repositories (from which we borrowed code) for making the code public. </br>
[Gradient Episodic Memory](https://github.com/facebookresearch/GradientEpisodicMemory) </br>
[Gumbel Softmax VAE](https://github.com/YongfeiYan/Gumbel_Softmax_VAE) </br>
[VQ-VAE](https://github.com/deepmind/sonnet)</br>
[VQ-VAE-2](https://github.com/rosinality/vq-vae-2-pytorch)</br>
