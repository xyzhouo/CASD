
# Cross-Attention-Based-Style-Distribution

The source code for our paper "[Cross Attention Based Style Distribution for Controllable Person Image Synthesis](https://arxiv.org/abs/2208.00712)" (**ECCV2022**).

<p align='center'>  
  <img src='https://github.com/xyzhouo/CASD/blob/main/head_img3_00.png' width='1000'/>
</p>


## Installation

#### Requirements

- Python 3
- PyTorch 1.7.0
- CUDA 10.2

#### Conda Installation

``` bash
# 1. Create a conda virtual environment.
conda create -n CASD python=3.6
conda activate CASD
conda install -c pytorch pytorch=1.7.0 torchvision=0.8.0 cudatoolkit=10.2

# 2. Install other dependencies.
pip install -r requirements.txt
```


### Data Preperation

The dataset structure is recommended as:
```
+—dataset
|   +—fashion
|       +--train (person images in 'train.lst')
|       +--test (person images in 'test.lst')
|       +--train_resize (resized person images in 'train.lst')
|       +--test_resize (resized person images in 'test.lst')
|       +--trainK(keypoints of person images)
|       +--testK(keypoints of person images)
|       +—semantic_merge3(semantic masks of person images)
|   +—fashion-resize-pairs-train.csv
|   +—fashion-resize-pairs-test.csv
|   +—fasion-resize-annotation-pairs-train.csv
|   +—fasion-resize-annotation-pairs-test.csv
|   +—train.lst
|   +—test.lst
|   +—vgg19-dcbb9e9d.pth
|   +—vgg_conv.pth
|   +—vgg.pth
...
```


1. Person images

- Download `img_highres.zip` of the DeepFashion Dataset from [In-shop Clothes Retrieval Benchmark](https://drive.google.com/drive/folders/0B7EVK8r0v71pYkd5TzBiclMzR00). 

- Unzip `img_highres.zip`. You will need to ask for password from the [dataset maintainers](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html). Then put the obtained folder **img_highres** under the `./dataset/fashion` directory. 

- Download train/test key points annotations and the train/test pairs from [Google Drive](https://drive.google.com/drive/folders/1qGRZUJY7QipLRDNQ0lhCubDPsJxmX2jK?usp=sharing) including **fashion-resize-pairs-train.csv**, **fashion-resize-pairs-test.csv**, **fashion-resize-annotation-train.csv**, **fashion-resize-annotation-test.csv,** **train.lst**, **test.lst**. Put these files under the  `./dataset/fashion` directory.

- Run the following code to split the train/test dataset.

  ```bash
  python tool/generate_fashion_datasets.py
  ```
  
- Run the following code to resize the train/test dataset.

  ```bash
  python tool/resize_fashion.py
  ``` 
  
  
2. Keypoints files

- Generate the pose heatmaps. Launch
  ```bash
  python tool/generate_pose_map_fashion.py
  ```

3. Segmentation files
- Extract human segmentation results from existing human parser (e.g. LIP_JPPNet). Our segmentation results ‘semantic_merge3’ are provided in [Google Drive](https://drive.google.com/drive/folders/1qGRZUJY7QipLRDNQ0lhCubDPsJxmX2jK?usp=sharing). Put it under the ```./dataset/fashion``` directory.


### Training

```bash
python train.py --dataroot ./dataset/fashion --dirSem ./dataset/fashion --pairLst ./dataset/fashion/fashion-resize-pairs-train.csv --name CASD_test --batchSize 16 --gpu_ids 0,1 --which_model_netG CASD --checkpoints_dir ./checkpoints
```
The models are save in `./checkpoints`. 

### Testing
Download our pretrained model from [Google Drive](https://drive.google.com/drive/folders/1qGRZUJY7QipLRDNQ0lhCubDPsJxmX2jK?usp=sharing). Put the obtained checkpoints under `./checkpoints/CASD_test`. Modify your data path and launch
```bash
python test.py --dataroot ./dataset/fashion --dirSem ./dataset/fashion --pairLst ./dataset/fashion/fashion-resize-pairs-test.csv --checkpoints_dir ./checkpoints --results_dir ./results --name CASD_test --phase test  --batchSize 1  --gpu_ids 0,0 --which_model_netG CASD --which_epoch 1000
```
The result images are save in `./results`. 

## Citation
If you use this code for your research, please cite
```
@article{zhou2022casd,
  title={Cross Attention Based Style Distribution for Controllable Person Image Synthesis},
  author={Zhou, Xinyue and Yin, Mingyu and Chen, Xinyuan and Sun, Li and Gao, Changxin and Li, Qingli},
  journal={arXiv preprint arXiv:2208.00712},
  year={2022}
}
```


