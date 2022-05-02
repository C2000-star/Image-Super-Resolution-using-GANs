# ESRGAN 

## Training and Testing

### Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

### Configuration File
The configuration file is present in the directory:
```
/configs/*.yaml
```

```python
# general setting
batch_size: 16
input_size: 32
gt_size: 128
ch_size: 3
scale: 4
log_dir: '/content/drive/MyDrive/ESRGAN'
pretrain_dir: '/content/drive/MyDrive/ESRGAN-MSE-500K-DIV2K'  # directory to load from at initial training
cycle_mse: True
# generator setting
network_G:
    name: 'RRDB'
    nf: 64
    nb: 23
    apply_noise: True
# discriminator setting
network_D:
    nf: 64

# dataset setting
train_dataset:
    path: '/content/drive/MyDrive/data/div2k_hr/DIV2K_train_HR'
    num_samples: 32208
    using_bin: True
    using_flip: True
    using_rot: True
    detect_blur: True
    buffer_size: 1024           # max size of buffer
    patch_per_image: 128        # number of patches to extract from each image
test_dataset:
    set5: './test data/Set5'
    set14: './test data/Set14'

# training setting
niter: 400000

lr_G: !!float 1e-4
lr_D: !!float 1e-4
lr_steps: [50000, 100000, 200000, 300000]
lr_rate: 0.5

adam_beta1_G: 0.9
adam_beta2_G: 0.99
adam_beta1_D: 0.9
adam_beta2_D: 0.99

w_pixel: !!float 1e-2
pixel_criterion: l1

w_feature: 1.0
feature_criterion: l1

w_gan: !!float 5e-3
gan_type: ragan  # gan | ragan
refgan: True       # provide reference image

save_steps: 5000

# logging settings
logging:
    psnr: True
    lpips: True
    ssim: True
    plot_samples: True
```

`cycle_mse`: use cycle-consistent content loss

`network_G/apply_noise`: provide random noise to the generator network

`train_dataset/detect_blur`: filter blurry images in the training dataset


Explanation of *config* files:
- `esrgan.yaml`: baseline ESRGAN (configuration(c))
- `use_noise.yaml`: +use noise (configuration(e))
- `cyclegan.yaml`: +cycle loss (configuration(f))

### Training
The training process is divided into two parts;
pretraining the model with pixel-wise loss, and training the pretrained PSNR model with ESRGAN loss.

#### Pretrain PSNR
Pretrain the PSNR RDDB model.
```bash
python train_psnr.py --cfg_path="./configs/psnr.yaml" --gpu=0
```

#### ESRGAN
Train the ESRGAN model with the pretrain PSNR model.
```bash
python train_esrgan.py --cfg_path="./configs/esrgan.yaml" --gpu=0
```
The DIV2K dataset is available [here](https://drive.google.com/drive/folders/1UXo1IreQYdgij3BURJ7T69TyCE2mk64r?usp=sharing)

## Evaluation

```
python test.py --model=weights/ESRGAN-cyclemixing --gpu=0 --img_path=photo/baby.png --down_up=True --scale=4(optional)
```
When the `down_up` option is `True`, the image will be arbitrarily downsampled and processed through the network. For real use cases, the option must be marked `False` for the model to upsample the image.  
## Pre-trained models and logs

Model `.tb` files can be downloded [here](https://drive.google.com/drive/folders/1an_riZB7vcyzHj3mZcRWaLueuBzfR0do?usp=sharing)

Four trained models are included in the repository in `./weights/*`.

## Results

The colab notebook with the required implementation and results is in the folder `\notebooks\`

# StyleGAN for super resolution

### Super Resolution
Given a low-resolution input image, we generate a corresponding high-resolution image. As this too is an ambiguous task, we can use style-mixing to produce several plausible results.


## Getting Started
### Prerequisites
- Linux or macOS
- NVIDIA GPU + CUDA CuDNN (CPU may be possible with some modifications, but is not inherently supported)
- Python 2 or 3

- Dependencies:  
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/). 
All dependencies for defining the environment are provided in `environment/psp_env.yaml`.

### Inference Notebook
To help visualize the StyleGAN framework on multiple tasks and to help you get started, we provide a Jupyter notebook found in `notebooks/inference_playground.ipynb` that allows one to visualize the various applications of pSp.   
The notebook will download the necessary pretrained models and run inference on the images found in `notebooks/test data`.  
For the tasks of conditional image synthesis and super resolution, the notebook also demonstrates pSp's ability to perform multi-modal synthesis using 
style-mixing. 

### Pretrained Models


If you wish to use one of the pretrained models for training or inference, you may do so using the flag `--checkpoint_path`.

By default, we assume that all auxiliary models are downloaded and saved to the directory `pretrained_models`. However, you may use your own paths by changing the necessary values in `configs/path_configs.py`. 

## Training
### Preparing your Data
- Currently, we provide support for numerous datasets and experiments (encoding, frontalization, etc.).
    - Refer to `configs/paths_config.py` to define the necessary data paths and model paths for training and evaluation. 
    - Refer to `configs/transforms_config.py` for the transforms defined for each dataset/experiment. 
    - Finally, refer to `configs/data_configs.py` for the source/target data paths for the train and test sets
      as well as the transforms.
- If you wish to experiment with your own dataset, you can simply make the necessary adjustments in 
    1. `data_configs.py` to define your data paths.
    2. `transforms_configs.py` to define your own data transforms.
    
``` 
dataset_paths = {
    'celeba_test': '/path/to/CelebAMask-HQ/test_img',
}
```
The transforms for the experiment are defined in the class `EncodeTransforms` in `configs/transforms_config.py`.   
Finally, in `configs/data_configs.py`, we define:
``` 
DATASETS = {
   'ffhq_encode': {
        'transforms': transforms_config.EncodeTransforms,
        'train_source_root': dataset_paths['ffhq'],
        'train_target_root': dataset_paths['ffhq'],
        'test_source_root': dataset_paths['celeba_test'],
        'test_target_root': dataset_paths['celeba_test'],
    },
}
``` 
When defining our datasets, we will take the values in the above dictionary.

### Training StyleGAN
The main training script can be found in `scripts/train.py`.   
Intermediate training results are saved to `opts.exp_dir`. This includes checkpoints, train outputs, and test outputs.  
Additionally, if you have tensorboard installed, you can visualize tensorboard logs in `opts.exp_dir/logs`.

#### **Training the pSp Encoder**
```
python scripts/train.py \
--dataset_type=ffhq_encode \
--exp_dir=/path/to/experiment \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=2500 \
--save_interval=5000 \
--encoder_type=GradualStyleEncoder \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=1 \
--id_lambda=0.1
```

#### **Super Resolution**
``` 
python scripts/train.py \
--dataset_type=celebs_super_resolution \
--exp_dir=/path/to/experiment \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=2500 \
--save_interval=5000 \
--encoder_type=GradualStyleEncoder \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=1 \
--id_lambda=0.1 \
--w_norm_lambda=0.005 \
--resize_factors=1,2,4,8,16,32
```