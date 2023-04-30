# SimSiam Logo Identifier

A computer vision network for brand logo identification based on the SimSiam neural network.
This network was created to satisfy the CSE 586 Computer Vision II class project requirement.

<p align="center">
    <img width="400" alt="simsiam" src="https://user-images.githubusercontent.com/2420753/118343499-4c410100-b4de-11eb-9313-d49e65440a7e.png">
</p>

This PyTorch implementation is based on the [SimSiam paper](https://arxiv.org/abs/2011.10566) by Chen, et al.

## Datasets
- [Car Brand Logos Kaggle dataset](https://www.kaggle.com/datasets/volkandl/car-brand-logos?resource=download) - a dataset containing 2,913 images of 8 car logos. Each brand has 300~350 training photos and ~50 test photos.

<div style="display: flex;">
<img src="https://storage.googleapis.com/kagglesdsdata/datasets/1451197/2399898/Car_Brand_Logos/Train/hyundai/10836079247410.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20230428%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230428T045932Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=a300b8557a4d8ccd1f1b99624b4ac4028e1a2bf30b719ad8321cd5801e6a4a50a58120fbe3ecca26bc3a55efea27286e01b921cdbaabdcf2b6075396a630a347fc506aec31a5b34c169f927619d980a4ba4a7f7715feec8f1aa261c530840b5a7d95802105ae2b16954c6e04f2055014ca816ef41daf87f7cd9b7d3853f90fdf5581c9c8b79ed8100700b2f20a93fd62ff7aeb3adb49a1cf681e0c2aeaca26dd5c50db88c720ac1a7724a1ff95f4b0d6bb7a512a037d693afa59f52eea34dc446d57a86445389c1bdc4a2c74599cc2f196e1930a5b4b7237570969d78dc4d7af3d44760f4969f646b16ca7a2e2b1be06f18a6679a2bc5af10c101bf99f4db7b3" alt="Hyundai logo" style="width: 200px; height: auto; margin-left: 100px;">

<img src="https://storage.googleapis.com/kagglesdsdata/datasets/1451197/2399898/Car_Brand_Logos/Train/toyota/0_org_zoom.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20230428%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230428T210036Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=2161b0fcaa33b237209f8024d6f22ec7242ccb4ba1d9637ab0d1741fc83c46c1eef1d46a34fefc9272319362571fee536992a538ee3590c84dd23a7d163d9e5d9866e3171fc596036a52478e2d36f87e8966269dbe5d0e5d555c19f384ec5059a37aac455357b6d744a143aef4df9cbf46c1dc654782cee94bf3839d7a65774b233b77e362c8b306abcbbe80b63a14ca17b0f9a6b7fb8eee8e1d7416d24c57c89fb70eb785f8dbd242733534659781966a779ec37a40febfe1f26852f2c4e1886ebd58c9f9ec7d8309f40a1148c26b2ec3e0176e478c69bb87c6ebff60f00cd96e6a4223a7e58ab82a3011f03c42d6ed6f0aa6bad7306b2773b60d5d629290c6" alt="Toyota logo" style="width: 200px; height: auto; margin-bottom: 20px">

<img src="https://storage.googleapis.com/kagglesdsdata/datasets/1451197/2399898/Car_Brand_Logos/Train/volkswagen/0_org_zoom.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20230428%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230428T210057Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=9f3ae0faa185c7e9e893b652d98140da5c225705d59224a9cb1f9d00b799a0bdef698eefbb2a96d4953e1a6176ef16b7f4c27e8ce9c5f5624ab82b997a168eb85749d758dc37304ce90695c50c12b55e5c9f86219baf6910bcdd00d934a4bb1aa34f2eeb38685b4b50ad03506378aaf89e14523ed2791ff9965ec7c56a55ba080088c4761de408f4ce665bc52de492751da8fb71d18557f7d171a958fdf01e087daca3575e4c85f3fdaa34fc8ef93bd7c2b12e8e9fdf693c19eec8263dd68c0dd92500137c26c571808434be488af69fcd7a8a40644e0286343322de1df8ccd3f8331c5e3cfe72ccbef97409f238b41a29d2185c73cd52f06109e595841d0dfa" alt="Volkswagen logo" style="width: 200px; height: auto;">
</div>


- [FlickerLogos Dataset](https://www.uni-augsburg.de/en/fakultaet/fai/informatik/prof/mmc/research/datensatze/flickrlogos/) - a brand logos dataset that contains 8,240 images of 32 logo classes from different industries.

<img src="https://assets.uni-augsburg.de/media/filer_public/75/74/75741202-144e-438e-a780-536d64b3e37a/2769032620.jpg" alt="DHL logo" style="width: 500px; height: auto; margin-left: 100px;">

## How to use
`Python simsiam_logo_training.py location-of-new-dataset --resume location-of-pretrained-network-checkpoint`
`Python simsiam_flickr_training.py --resume location-of-pretrained-network-checkpoint`
`Python simsiam_logo_classification.py location-of-new-dataset --pretrained location-of-pretrained-network-checkpoint`
`Python simsiam_flickr_classification.py --pretrained location-of-pretrained-network-checkpoint`

## License
This project is under the CC-BY-NC 4.0 license (same as the [SimSiam paper code](https://github.com/facebookresearch/simsiam)).

## Contributors
Dylan Knowles and Akash Kumar
