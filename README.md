# UNet_Simplified
	Implementation of the UNet CNN model for semantic segmentation as 
    devised by Ronneberger et al. 2015 [1]. An additional hyperparameter, 
    'depth', has been added to procedurally control the number of convolution
    blocks in the contracting path and deconvolution blocks in the
    expanding path. The number of layers and the number of weights
    increases with depth. As implemented here, UpSampling2D + Conv2D
    is used for upsampling in the expanding path instead of Conv2DTranspose
    in order to reduce checkerboard-artifacts [2].
	
	This implementation has been designed with a focus on multilevel-segmentation
	of light-microscopic images of the slime mold Physarum polycephalum [3], and has only
	been tested with this type of image. A typical scenario with three segmentation levels
	is shown below. It should be applicable to other types of images, but might require
	adaptation of the preprocessing steps taken in the training / segmentation functions,
	and possibly different data augmentation steps.
	
![alt text](https://github.com/adrianfessel/UNet_Simplified/blob/main/overlay.png?raw=true)
	
# Requirements
	The implementation is based on tensorflow/keras and as any cnn model, works best if executed
	on a gpu. Other packages required include os, numpy, matplotlib, opencv, PIL, sklearn and tqdm.
	
# Specifications for training data
	images : any generic grayscale image specification should work (color images will be converted 
		to grayscale during image reading)
	labels : grayscale or color image, where each discrete gray- or color value corresponds to one class
	
# Contents
	model.py - specifies the cnn model
	training.py - contains an example training procedure as well as routines for image reading
		and a data generator for image/label map pairs
	segmentation.py - a class for automatic segmentation of directories of images
	
# References
	[1] : https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28
	[2] : https://distill.pub/2016/deconv-checkerboard/
	[3] : https://iopscience.iop.org/article/10.1088/1361-6463/ab866c
