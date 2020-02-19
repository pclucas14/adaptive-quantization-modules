## LiDAR experiments

To reproduce the SNNRSME in the paper, along with the displayed sampled please run 

```
bash reproduce_lidar.sh
```

By default, the code will save lidars from the buffer, as well as reconstructions from the test set in `../lidars`. 

### Displaying the results

I use the `Mayavi` to plot the results. I personally run the code on a server, so before anything else I need to send the point clouds to my local machine (not required if your server supports GUI). Running the following command will a folder on your server accessible from your local machine.

```
sshfs <username>@server.something.ca:<path_on_remote_machine> <path_on_local_machine>
```

For example, if I'm running experiments on the `beluga.computecanada.ca` server, that my username is `johndoe`, that the point clouds are saved on `/home/johndoe/modular-vqvae/lidars` and that I want to link the previous directory to `/home/Desktop/my_tmp_dir`, I would run the following command

```
sshfs johndoe@beluga.computecanada.ca:/home/johndoe/modular-vqvae/lidars /home/Desktop/my_tmp_dir
```

The actual code to display the point clouds in in `utils/kitti_utils.py` (note that you can simply copy this file inside the directory where the point clouds are located when doing sshfs, and run it on your local machine). By default, if you pass a filepath as 1st argument to `kitti_utils.py` it will display the first 4 point clouds. For example, 

```
python kitti_utils.py 'DSkitti_NB3_RTH[7e-05, 7e-05, 7e-05]_Comp8.00^16.00^32.00^_Coef1.10_3453_test12_11'
```

will display the test set recontructions we get after the 3rd epoch

| |
|:-------------------------:|
|32x compression |
|<img width="1604" src="https://github.com/pclucas14/modular-vqvae/blob/master/lidar/imgs/1.png"> |
|16x compression |
|<img width="1604" src="https://github.com/pclucas14/modular-vqvae/blob/master/lidar/imgs/2.png"> |
|8x compression |
|<img width="1604" src="https://github.com/pclucas14/modular-vqvae/blob/master/lidar/imgs/3.png"> |
|original LiDAR |
|<img width="1604" src="https://github.com/pclucas14/modular-vqvae/blob/master/lidar/imgs/4.png"> |


