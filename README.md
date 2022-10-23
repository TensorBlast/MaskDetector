# MaskDetector
Detects human faces with/without masks in a static image

## Run App
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ankit255/MaskDetector/HEAD?urlpath=%2Fvoila%2Frender%2FMaskDetector.ipynb)

## Dataset
This dataset consists of 4095 images belonging to two classes:

with_mask: 2165 images

without_mask: 1930 images

## Model
The model is based on Inception V3 pretrained on ImageNet dataset and finetuned on the dataset. 
OpenCV pre-trained Haar-cascade Detection classifier is used to extract faces as regions of interest(ROI) to feed into the trained Inception V3 model.
