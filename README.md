# Doge Detector

## What dog is this?

This is a project developed as a final project at the Data Science bootcamp organised by Sages.

The goal of the project is to recognize the breed of a dog from a photo taken by the user. The main idea is to prepare a model using transfer learning from other already trained models and train it on [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/). To get more accurate results, the first step of the picture processing is the detection of the objects. For this purpose, the [YOLO](https://pjreddie.com/darknet/yolo/) system is used. Keras wrapper for YOLO can be found [here](https://github.com/xiaochus/YOLOv3)

As base models for feature extraction, two well know CNN are used: **ResNet** and **Inception**. Each of them is used to extract the features of the picture, which are then used as an input for the simple convolutional network, which is responsible for classifying the image to one of 120 classes (dogs' breeds)

## Dataset

[Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) contains images of 120 breeds of dogs from around the world. This dataset has been built using images and annotation from ImageNet for the task of fine-grained image categorization. Files need for the project to work are:

- [Images](http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar) (757MB) 

- [Annotations](http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar) (21MB)

- [Lists](http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar), with train/test splits (0.5MB)

## Environment

Environment can be recreated using yaml file available in the repo. To recreate environemt use following commands in terminal:

```shell
$ conda env create -f doge_detector.yml
```

When the packages are installed, you can change the env to the one created above:

```shell
#osx
$ conda activate doge_detector

# or

#linux
$ source activate doge_detector
```

## Usage

Run jupyter notebook inside doge_detector environment (the one created in **Environment** section).

```shell
$ jupyter notebook
```

Then open a notebook and have fun :)

## Accuracy

The total accuracy of the models are:

```
resnet_match       0.808508
inception_match    0.898601
```

## Example

![Example prediction for the file outside of the dataset](https://i.imgur.com/Ly1sC5O.png)




