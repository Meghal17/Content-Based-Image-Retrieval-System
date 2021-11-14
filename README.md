# Content Based Image Retrieval System (CBIR)

This is an attempt to build a CBIR system for images. You can use your 
own dataset of images for which you wish to build a CBIR system. 

## Working

The system uses a Convolutional Neural Network (ResNet-50) to extract the features
of all the images in the dataset. Then these features are clustered by k-means 
clustering (supervised, meaning you know the number of different classes of images 
in the dataset. Finally, any number of similar images can be obtained using a query 
image.

## To use this system:

- Copy the dataset of all the images (all images belonging to different classes 
in single directory).
- Use Config.txt to configure the path settings for dataset, clustered data, and models.
- run ```Cluster_Data.py``` to extract features of all images and cluster the images based on 
the extracted features.
- run ```Search_Images.py``` to retrieve desired number of similar images.
