# AHA: Human-Assisted Out-of-Distribution Generalization and Detection

This codebase provides a Pytorch implementation for the paper: [AHA: Human-Assisted Out-of-Distribution Generalization and Detection]() by Haoyue Bai, Jifan Zhang, Robert Nowak.


### Abstract

Modern machine learning models deployed often encounter distribution shifts in real-world applications, manifesting as covariate or semantic out-of-distribution (OOD) shifts. These shifts give rise to challenges in OOD generalization and OOD detection. This paper introduces a novel, integrated approach AHA (Adaptive Human-Assisted OOD learning) to simultaneously address both OOD generalization and detection through a human-assisted framework by labeling data in the wild. Our approach strategically labels examples within a novel maximum disambiguation region, where the number of semantic and covariate OOD data roughly equalizes. By labeling within this region, we can maximally disambiguate the two types of OOD data, thereby maximizing the utility of the fixed labeling budget. Our algorithm first utilizes a noisy binary search algorithm that identifies the maximal disambiguation region with high probability. The algorithm then continues with annotating inside the identified labeling region, reaping the full benefit of human feedback. Extensive experiments validate the efficacy of our framework. We observed that with only a few hundred human annotations, our method significantly outperforms existing state-of-the-art methods that do not involve human assistance, in both OOD generalization and OOD detection.


## Quick Start

### Data Preparation
In this work, we evaluate the OOD generalization and detection performance over a range of environmental discrepancies such as domains, image corruptions, and perturbations. 

Download the data in the folder

```
./datasets
```



#### CIFAR-10 & CIFAR-10-C

- Create a folder named `cifar-10/` and a folder `cifar-10-c/` under `$datasets`.
- Download the dataset from the [CIFAR-10](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) and extract the training and validation sets to `$DATA/cifar-10/`.
- Refer the dataset from the [CIFAR-10-C](https://arxiv.org/abs/1903.12261) and extract the training and test sets to `$DATA/cifar-10-c/`. The directory structure should look like


The corrupted CIFAR-10 dataset can be downloaded via the link:
```
wget https://drive.google.com/drive/u/0/folders/1JcI8UMBpdMffzCe-dqrzXA9bSaEGItzo
```


```
cifar-10/
|–– train/ 
|–– val/
cifar-10-c/
|–– CorCIFAR10_train/ 
|–– CorCIFAR10_test/
```


Here are links for the less common semantic OOD datasets regarding CIFAR benchmark used in the paper: 
[Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/),
[Places365](http://places2.csail.mit.edu/download.html), 
[LSUN](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz),
[LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz),
[iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz).

For example, run the following commands in the **root** directory to download **LSUN-C**:
```
cd data/LSUN
wget https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz
tar -xvzf LSUN.tar.gz
```


#### ImageNet-100 & ImageNet-100-C

- Create a folder named `imagenet-100/` and a folder `imagenet-100-c/` under `$datasets`.
- Create `images/` under `imagenet-100/` and `imagenet-100-c/.
- Download the dataset from the [ImageNet-100](https://image-net.org/index.php](https://github.com/deeplearning-wisc/MCM/tree/main) and extract the training and validation sets to `$DATA/imagenet-100/images`.
- Download the dataset from the [ImageNet-100-C](https://arxiv.org/abs/1903.12261) and extract the training and validation sets to `$DATA/imagenet-100-c/images`. The directory structure should look like

```
imagenet-100/
|–– images/
|   |–– train/ # contains 100 folders like n01440764, n01443537, etc.
|   |–– val/
imagenet-100-c/
|–– images/
|   |–– train/ 
|   |–– val/
```

For large-scale experiments, we use iNaturalist as the semantic OOD dataset. We have sampled 10,000 images from the selected concepts for iNaturalist,
which can be downloaded via the following link:
```
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
```


## Training and Evaluation 

### Pretrained models

# Pretrained models

You can find the pretrained models in 

```
./checkpoints/Resnet34_vanilla.pt
```



**Demo** 

We provide sample scripts to run the code. Feel free to modify the hyperparameters and training configurations.

```
bash run.sh
```


### Citation

If you find our work useful, please consider citing our paper:


### Further discussions
For more discussions on the method and extensions, feel free to drop an email at hbai39@wisc.edu
