# VQ-GAN + CLIP

Repository stems from: [Original colab](https://colab.research.google.com/drive/1Zx38waUmaF3bk8ejjB21HzGiuAOnX4PD?usp=sharing)
 
 
Additional upscaling using [ESRGAN](https://github.com/xinntao/ESRGAN)

Very nice introduction to the technique: [Alien Dreams](https://ml.berkeley.edu/blog/posts/clip-art/)
 
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

## Run
 ```
python CLIP_VQGAN.py -texts your_text_prompt
 ```
**Additional run option**
- _width_ - Image width
- _height_ - Image height
- _model_ - Used pretrained model for VQ-GAN
- _display_int_ - Display interval during generation of the image
- _init_image_ - Starting image instead of random noise
- _target images_ - Target images instead of text prompt
- _seed_ - Random seed
- _max_iterations_ - Maximum number of optimization iterations
- _make_video_ - Possibility of making video from genrated images
- _upscale_ - Possibility to 4x upscale images 

## Additional information

Work in progress...
