# SimSiam Logo Identifier

A computer vision network for brand logo identification based on the SimSiam neural network.
This network was created to satisfy the CSE 586 Computer Vision II class project requirement.

<p align="center">
    <img width="400" alt="simsiam" src="https://user-images.githubusercontent.com/2420753/118343499-4c410100-b4de-11eb-9313-d49e65440a7e.png">
</p>

This PyTorch implementation is based on the [SimSiam paper](https://arxiv.org/abs/2011.10566) by Chen, et al.

## Datasets Used
- [Car Brand Logos Kaggle dataset](https://www.kaggle.com/datasets/volkandl/car-brand-logos?resource=download) - a dataset containing 2,913 images of 8 car logos. Each brand has 300~350 training photos and ~50 test photos.

<div style="display: flex;">
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Hyundai_Motor_Company_logo.svg/1920px-Hyundai_Motor_Company_logo.svg.png" alt="Hyundai logo" style="width: 200px; height: auto;">
&ensp;&ensp;
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Toyota.svg/1920px-Toyota.svg.png" alt="Toyota logo" style="width: 200px; height: auto;">
&ensp;
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/75/Lexus.svg/1920px-Lexus.svg.png" alt="Lexus logo" style="width: 200px; height: auto;">
</div>
&ensp;

- [FlickerLogos Dataset](https://www.uni-augsburg.de/en/fakultaet/fai/informatik/prof/mmc/research/datensatze/flickrlogos/) - a brand logos dataset that contains 8,240 images of 32 logo classes from various industries.
<div style="display: flex;">
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;
<img src="https://assets.uni-augsburg.de/media/filer_public/75/74/75741202-144e-438e-a780-536d64b3e37a/2769032620.jpg" alt="DHL logo" style="width: 500px; height: auto;">
</div>

## How to Use
- To train the network on your own dataset, run:  
`python simsiam_logo_training.py [path/to/your/dataset/] --resume [path/to/pretrained/network/checkpoint/]` OR    
`python simsiam_flickr_training.py [path/to/your/dataset/] --resume [path/to/pretrained/network/checkpoint/]`
- To classify images in your dataset, run:  
`python simsiam_logo_classification.py [path/to/your/dataset/] --pretrained [path/to/pretrained/network/checkpoint/]` OR  
`python simsiam_flickr_classification.py [path/to/your/dataset/] --pretrained [path/to/pretrained/network/checkpoint/]`

## License
This project is under the CC-BY-NC 4.0 license (same as the [SimSiam paper code](https://github.com/facebookresearch/simsiam)).

## Contributors
Dylan Knowles and Akash Kumar
