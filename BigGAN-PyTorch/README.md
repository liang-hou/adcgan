# ADC-GAN
The author's officially PyTorch ADC-GAN implementation based on the [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch) repository.

## Usage
You will need:

- [PyTorch](https://PyTorch.org/), version 1.6.0
- sklearn, tqdm, numpy, scipy, and h5py
- The Tiny-ImageNet dataset (if needed) following the below structure
```
tiny_imagenet
    train
      cls0
        img0
        ...
      ...
    valid
      cls0
        img0
        ...
      ...
```


Before running methods on a dataset named `DATASET={C10,C100,TI200}`, please run the following command to prepare the statistics for calculating FID and intra-FID.
```
python calculate_inception_moments.py --dataset DATASET --data_root data
```

To run `METHOD={pdgan, acgan, amgan, tacgan, adcgan}` on `DATASET={cifar10, cifar100,tinyimagenet}`, please run
```
bash scripts/luanch_[DATASET]_ema_[METHOD].sh
```

To evaluate the intra-FID (generation quality) and Accuracy (representation quality) results, please refer to `test.py`

## Acknowledgments
The code is developed based on [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch).
