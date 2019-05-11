# Faster R-CNN Inference with Minimum Implementation

A minimum implementation based on [mask-rcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark), 
aiming at easier understanding of the 2-stage object detection framework using Faster R-CNN as an example.
See the [Pipeline](PIPELINE.md) to understand how it works!

Let's make the framework more specific, and easier to understand!

## What's New 

- verified inference of Faster R-CNN C4 using pretrained weights

## Inference
Easy enough! 

```bash
python tools/demo.py --image {image file path} 
```


## Installation

### Docker

Build image with defaults (`CUDA=9.0`, `CUDNN=7`, `FORCE_CUDA=1`):

    nvidia-docker build -t faster_rcnn_minimum docker/
    
Build image with other CUDA and CUDNN versions:

    nvidia-docker build -t faster_rcnn_minimum --build-arg CUDA=9.2 --build-arg CUDNN=7 docker/

### conda

```bash

conda create --name faster_rcnn_minimum
conda activate faster_rcnn_minimum

conda install ipython
pip install ninja yacs cython matplotlib tqdm
conda install -c pytorch pytorch-nightly torchvision cudatoolkit=9.0

python setup.py build develop
```

