# Convolutional Neural Networks

A Convolutiona Neural Network (CNN) is a specific type of feed-forward deep network. CNNs are inspired by the visual cortex of the brain. Some of the individual neurons of the visual cortex are activated only when edges of a particular orientation (e.g. vertical, horizontal) are viewed. These neurons are ordered together in a column and combine together to produce visual perception. Essentially, it is the idea that individual components of a system are specialized for specific tasks. 

[This video gives a great, easy to understand, explanation of CNNs](https://www.youtube.com/watch?v=JiN9p5vWHDY&list=PLjJh1vlSEYgvGod9wWiydumYl8hOXixNu&index=8). 

[Very useful resource](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)

[This is also a very helpful resource](http://cs231n.github.io/convolutional-networks/)

[As is this course on Kaggle](https://www.kaggle.com/dansbecker/intro-to-deep-learning-and-computer-vision)

A CNN has three types of layers: convolutional layers, activation (relu) layers, and pooling layers. In Keras, the convolution and activation layers are added at the same time. Many of the explanations will use images as an example, as that is the typically use for a CNN, but in the end we will actually be using the LB spectrum. 

## Convolutional Layer
The convolutional layer is the part of the CNN that does the 'heavy lifting'. It consists of a series of learnable filters (kernels). Each filter is set to a certain small size (e.g. 3x3 for a 2D BW image, or 1x3 for a 1D vector). During the pass through the layer, the filter is slid (convolved) across the width and height of the input. As the filter is slid, an activation or feature map is produced that gives the response of the filter at every spatial position. The network will learn filters that activate when they see a specific feature, such as edges if the input is an image. Each convolutional layer has multiple filters and each filter produces a separate activation or feature map. These activation maps are stacked together and passed to the next layer. 

For example, consider a 5 by 5 image (gree) which only has values 0 or 1 and a 3 by 3 filter (yellow):  

![image](https://ujwlkarn.files.wordpress.com/2016/07/screen-shot-2016-07-24-at-11-25-13-pm.png?w=127&h=115)  ![filter](https://ujwlkarn.files.wordpress.com/2016/07/screen-shot-2016-07-24-at-11-25-24-pm.png?w=74&h=64)

The convolution of the filter across the image would look like this:

![image](https://ujwlkarn.files.wordpress.com/2016/07/convolution_schematic.gif?w=268&h=196)

The filter is slide across the image by 1 pixel. At every position, the filter is first multiplied elementwise with the pixels of the image and then the resulting values are summed up to get a final integer which is then a single element of the output matrix (feature map). Different values in the filter will create a different output feature map. 

Here is a helpful graphic that shows the result of two different filters being convolved across an image:

![image](https://ujwlkarn.files.wordpress.com/2016/08/giphy.gif?w=748)


## Pooling Layer
