# VQ-GAN + CLIP

Repository stems from:

[Original colab](https://colab.research.google.com/drive/1Zx38waUmaF3bk8ejjB21HzGiuAOnX4PD?usp=sharing)
 
 
Additional upscaling using [ESRGAN](https://github.com/xinntao/ESRGAN)
 
 ## Installation
 
If you want to run the script on GPU, firstly install PyTorch with CUDA support!
 
 ```
git clone https://github.com/openai/CLIP 
git clone https://github.com/CompVis/taming-transformers 
git clone https://github.com/xinntao/ESRGAN
pip install ftfy 
pip install regex
pip install tqdm
pip install omegaconf
pip install pytorch-lightning
pip install kornia 
pip install einops 
pip install imageio-ffmpeg
pip install opencv-python
 ```

## Pretrained models

Copy pretrained models into _models/_

### vqgan_imagenet_f16_16384
[model](https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1)
[config](https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1)

Additional links to models - work in progress...

## Additional information

Work in progress...
