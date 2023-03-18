# SimSiam Logo Identifier

A computer vision neural network for brand logo identification.
This network was created to satisfy the CSE 586 Computer Vision II class project requirement.

<p align="center">
    <img width="400" alt="simsiam" src="https://user-images.githubusercontent.com/2420753/118343499-4c410100-b4de-11eb-9313-d49e65440a7e.png">
</p>

This PyTorch implementation is based on the [SimSiam paper](https://arxiv.org/abs/2011.10566):

## Datasets
- [Car Brand Logos Kaggle dataset](https://www.kaggle.com/datasets/volkandl/car-brand-logos?resource=download) - a dataset containing 2,913 images of 8 car logos. Each brand has 300~350 training photos and ~50 test photos.

<img src="https://storage.googleapis.com/kagglesdsdata/datasets/1451197/2399898/Car_Brand_Logos/Train/hyundai/10836079247410.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20230318%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230318T215132Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=aa179c174c0fb8969a74ead9875614dabb6f43af42b0136ecc3df2ec3061b6be242bcfe6faeb0aa5c616c9ebbde64b58893ca2b14aa87f90217f67ab85dfe33350b3bc8271c7d52c144afa307b5679b005507669593510bfa2dce2cedf968e06b086e0b8f58785e4d46ce86e81546ccbe327ed3e717fa4afa7d546029e63abb604d821dc954e628d2155e6b2c1f8d0446195b67923607ba01a3d54b2a9c08ebc414fac1457b7560b07ebaee0b20c5122d34b9e07c3bf0f4cfba5a68834cc0b84954950b0df713b0762a6ceffed4753348e9d9f5e2710d9b26031e50427a43b4f5472f4251c777d59697a10ad0e2e1f1b77de8cbf87f466f644074ed611cbfd79" alt="Hyundai logo" style="width: 140px; height: auto; margin-left: 100px; margin-bottom: -30px;">

<img src="https://storage.googleapis.com/kagglesdsdata/datasets/1451197/2399898/Car_Brand_Logos/Train/toyota/a13.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20230318%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230318T215200Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=2d55c481195c29e95aee958e82f3496cedddbf59ed2154a88399d5aa88623496c1ef2b6044f8d93f6adaa798f73e09d28840f46d8e703dc35018e51db33afa9108daedefeb3bfe349c414de3fcb202c5bcbe6705bf89b0b9ace712e10909e6d4c806d1316489977b0b8b9bcd4084356576d7acc2b244c0639fa474dbd4fc53fbd37dab16835be1da7a046769348b2ed855dd4b222ada8527054ef5c4252bdfd438931fbd23dd3956911e7127c180069276983e95ff7f0b3bde48875bd6c82cf6b10d9128ec0717a9d1e2725d9d0f09add831456edc454bfc098cab281b9497ef608a10b18b1d64de2b435009cd996e3b04bc74e1442cd2990dbf28c06b9c9e6d" alt="Toyota logo" style="width: 200px; height: auto;">

<img src="https://storage.googleapis.com/kagglesdsdata/datasets/1451197/2399898/Car_Brand_Logos/Train/volkswagen/0dyrtnsy-1-627x425.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20230318%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230318T215542Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=0e87705f678a318f0f63eadce0dbe497bb439532a3569b7957eec04eabd5683b3e8e37abb05ad2b8347c1e4d110427d31a11929df9bc3cd16413f982c9d6f96e538f5d196b3f80f3ef3527e16b24100d82ec95194d0102cd080d53c1238922c61da3303f7801ca16db2d0034c19e2acead125fcc2b25a94ff44370d5910bc49ce2306b2359a8afd073c95b0dc7ad872010b500c1674de5885e568702fd743f077636f5d5e98f82f611bfb6f892d8ef98d1112dd481c705944a51e74480a7305a6e36428db355d4394fe485554049769c58259d28a8caad6d67fb98e583994a2278faa7afb288470a065804490f644903b90a9317d84ce6043c28816b4b087518" alt="Volkswagen logo" style="width: 200px; height: auto;">

  

- [FlickerLogos Dataset](https://www.uni-augsburg.de/en/fakultaet/fai/informatik/prof/mmc/research/datensatze/flickrlogos/) - a brand logos dataset that contains 8,240 images of 32 logo classes.

<img src="https://assets.uni-augsburg.de/media/filer_public/75/74/75741202-144e-438e-a780-536d64b3e37a/2769032620.jpg" alt="DHL logo" style="width: 500px; height: auto; margin-left: 100px;">


## License
This project is under the CC-BY-NC 4.0 license (same as the SimSiam paper code).

## Contributors
Dylan Knowles and Akash Kumar
