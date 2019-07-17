# GAN steerability
[Project Page](https://ali-design.github.io/gan_steerability/) |  [Paper](https://arxiv.org/abs/1907.07171) 

<img src='img/teaser.jpeg' width=600>  

On the "steerability" of generative adversarial networks.\
[Ali Jahanian](http://people.csail.mit.edu/jahanian)\*, [Lucy Chai](http://people.csail.mit.edu/lrchai/)\*, [Phillip Isola](http://web.mit.edu/phillipi/)

## Prerequisites
- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Setup
- Clone this repo:
```bash
git clone https://github.com/ali-design/gan_steerability.git
```

- Install dependencies:
	- we provide a Conda `environment.yml` file listing the dependencies. You can create a Conda environment with the dependencies using:
```bash
conda env create -f environment.yml
```

- Download resources:
	- we provide a script for downloading associated resources (e.g. stylegan). Fetch these by running:
```bash
bash  resources/download_resources.sh
```

## Training walks
- The current implementation covers these variants:
	- models: biggan, stylegan
	- transforms: color, colorlab, shiftx, shifty, zoom, rotate2d, rotate3d
	- walk_type: linear, NNz
	- losses: l2, lpips
- Some examples of commands for training walks:
```bash
# train a biggan NN walk for shiftx with lpips loss
python train.py --model biggan --transform shiftx --num_samples 20000 --learning_rate 0.0001 \
	--walk_type NNz --loss lpips --gpu 0 --eps 25 --num_steps 5

# train a stylegan linear walk with l2 loss using the w latent space
python train.py --model stylegan --transform color --num_samples 2000 --learning_rate 0.0001 \
	--walk_type linear --loss l2 --gpu 0 --latent w --model_save_freq 100
```
- Alternatively you can train using a config yml file, for example:
```bash
python train.py --config_file config/biggan_color_linear.yml
```

- Each training run will save a config file called `opt.yml` in the same directory as the weights, which can be used for rerunning experiments with the same settings. You may need to use the flag `--overwrite_config` for overwriting existing weights and config files. 

- Run `python train.py -h` to list available training options


## Visualize walks

- Run `python vis_image.py -h` to list available visualization options. The key things to provide are a model checkpoint and a config yml file. For example:

```bash
python vis_image.py \
	models_pretrained/biggan_zoom_linear_lr0.0001_l2/model_20000_final.ckpt \
	models_pretrained/biggan_zoom_linear_lr0.0001_l2/opt.yml \
	--gpu 0 --num_samples 50 --noise_seed 20 --truncation 0.5 --category 207

python vis_image.py \
        models_pretrained/stylegan_color_linear_lr0.0001_l2_cats_w/model_2000_final.ckpt \
        models_pretrained/stylegan_color_linear_lr0.0001_l2_cats_w/opt.yml \
        --gpu 1 --num_samples 10 --noise_seed 20 
```

- We added some pretrained weights in the `./models_pretrained`, but you can also use the models you train yourself.

- By default this will save generated images to `<output_dir>/images` specified in the config yml, unless overwritten with the `--output_dir` option


## Notebooks

- We provide some examples of jupyter notebooks illustrating the full training pipeline. See [notebooks](./notebooks).
- It might be easiest to start here if you want to try your own transformations! The key things to modify are `get_target_np` and to scale alpha appropriately when feeding to graph.
- If using the provided conda environment, you'll need to add it to the jupyter kernel:
```bash
source activate gan_steerability
python -m ipykernel install --user --name gan_steerability
```

## Visualizing Distributions

These will be coming soon!

### Citation
If you use this code for your research, please cite our paper:
```
@article{gansteerability,
  title={On the "steerability" of generative adversarial networks},
  author={Jahanian, Ali and Chai, Lucy and Isola, Phillip},
  journal={arXiv preprint arXiv:1907.07171},
  year={2019}
}
```

