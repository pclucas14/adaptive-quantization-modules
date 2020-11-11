# Online Learned Continual Compression with Adaptive Quantization Modules (ICML 2020)
Stacking Quantization blocks for efficient lifelong online compression </br>
Code for reproducing all results in our paper  which can be found [here](https://arxiv.org/abs/1911.08019) </br>
You can find a quick demo on Google Colab [here](https://colab.research.google.com/drive/1bR5y0ko5QRPc6-_g8X5ok0XiiFcZVJtH?usp=sharing)


## (key) Requirements 
- Python 3.7
- Pytorch 1.4.0

## Structure
    ├── Common 
        ├── modular.py          # Module (QLayer) and Stacked modules (QStack). Includes most key ops, such as adaptive buffer        
        ├── quantize.py         # Discretization Ops (GumbelSoftmax, Vector/Tensor Quantization and Argmax Quantization)
        ├── model.py            # Encoder, Decoder, Classifier Blocks 
    ├── config                  # .yaml files specifying different AQM architectures and hyperparameters used in the paper 
    ├── Lidar
        ├── ....                # files to run LiDAR experiments 
    ├── Utils             
        ├── args.py             # Contains command-line args
        ├── buffer.py           # Basic buffer implementation. Handled raw and compressed representations
        ├── data.py             # CL datasets and dataloaders
        ├── utils.py            # Logging / Saving & Loading Models, Args, point cloud processing
        
    ├── gen_main.py             # files to run the offline classification (e.g. Imagenet) experiments 
    ├── eval.py                 # evaluation loops for drift, test acc / mse, and lidar
    ├── cls_main.py             # files to run the online classification (e.g. CIFAR) experiments
    
    ├── reproduce.txt           # All command and information to reproduce the results in the paper
        

## Acknowledgements 
We would like to thank authors of the following repositories (from which we borrowed code) for making the code public. </br>
[Gradient Episodic Memory](https://github.com/facebookresearch/GradientEpisodicMemory) </br>
[VQ-VAE](https://github.com/bshall/VectorQuantizedVAE) </br>
[VQ-VAE-2](https://github.com/rosinality/vq-vae-2-pytorch)</br>
[MIR](https://github.com/optimass/Maximally_Interfered_Retrieval)

## Contact
For any questions / comments / concerns, feel free to open an issue via github, or to send me an email at <br /> `lucas.page-caccia@mail.mcgill.ca`. <br />

We strongly believe in fully reproducible research. To that end, if you find any discrepancy between our code and the paper, please let us know, and we will make sure to address it.  <br />

Happy streaming compression :)

## Citation

If you find this code useful please cite us in your work.

```
@article{caccia2019online,
  title={Online Learned Continual Compression with Adaptive Quantization Modules},
  author={Caccia, Lucas and Belilovsky, Eugene and Caccia, Massimo and Pineau, Joelle},
  journal={Proceedings of the 37th International Conference on Machine Learning},
  year={2020}
}
```
