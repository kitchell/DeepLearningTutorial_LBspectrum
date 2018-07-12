# Convolutional Neural Networks

A Convolutional Neural Network (CNN) is a specific type of feed-forward deep network. CNNs are inspired by the visual cortex of the brain. Some of the individual neurons of the visual cortex are activated only when edges of a particular orientation (e.g. vertical, horizontal) are viewed. These neurons are ordered together in a column and combine together to produce visual perception. Essentially, it is the idea that individual components of a system are specialized for specific tasks. 

[This video gives a great, easy to understand, explanation of CNNs](https://www.youtube.com/watch?v=JiN9p5vWHDY&list=PLjJh1vlSEYgvGod9wWiydumYl8hOXixNu&index=8). 

[Very useful resource and source of the images used below](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)

[This is also a very helpful resource](http://cs231n.github.io/convolutional-networks/)

[As is this course on Kaggle](https://www.kaggle.com/dansbecker/intro-to-deep-learning-and-computer-vision)

A CNN has three types of layers: convolutional layers, activation (relu) layers, and pooling layers. In Keras, the convolution and activation layers can be added at the same time. CNNs are typically used for images. 

## Convolutional Layer
The convolutional layer is the part of the CNN that does the 'heavy lifting'. It consists of a series of learnable filters (kernels). Each filter is set to a certain small size (e.g. 3x3 for a 2D BW image, or 1x3 for a 1D vector). During the pass through the layer, the filter is slid (convolved) across the width and height of the input. As the filter is slid, an activation or feature map is produced that gives the response of the filter at every spatial position. The network will learn filters that activate when they see a specific feature, such as edges if the input is an image. Each convolutional layer has multiple filters and each filter produces a separate activation or feature map. These activation maps are stacked together and passed to the next layer. 

For example, consider a 5 by 5 image (green) which only has values 0 or 1 and a 3 by 3 filter (yellow):  

![image](https://ujwlkarn.files.wordpress.com/2016/07/screen-shot-2016-07-24-at-11-25-13-pm.png?w=127&h=115)  ![filter](https://ujwlkarn.files.wordpress.com/2016/07/screen-shot-2016-07-24-at-11-25-24-pm.png?w=74&h=64)

The convolution of the filter across the image would look like this:

![image](https://ujwlkarn.files.wordpress.com/2016/07/convolution_schematic.gif?w=268&h=196)

The filter is slid across the image by 1 pixel. At every position, the filter is first multiplied elementwise with the pixels of the image and then the resulting values are summed up to get a final integer which is then a single element of the output matrix (feature map). Different values in the filter will create a different output feature map. 

Here is a helpful graphic that shows the result of two different filters being convolved across an image:

![image](https://ujwlkarn.files.wordpress.com/2016/08/giphy.gif?w=400h=300)


In Keras, a convolutional layer is added by using a Conv1D (for 1D convolutions) or Conv2D (for 2D convolutions) layer:

```python
keras.layers.Conv1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

The arguments we care about for these layers are:
* filters - the number of filters used in the layer
* kernel_size - the size of the filters
* strides - typically = 1, maybe 2, the number of 'pixels'/'elements' the filter shifts over when convolving the image
* padding - the amount of empty zero padding around the image, sometimes it is helpful to border the image with 0s, default is no padding
* activation - we can combine the conv layer and relu layer in one step by setting this equal to 'relu'

## ReLU Layer

A ReLU layer is typically used after every convolution layer. In keras it can be added by setting the activation argument to 'relu'. ReLU stands for rectified linear unit and it is a non-linear operation defined by max(zero, input). Essentially it sets all negative 'pixels'/'elements' of the activation/feature maps to 0. Its purpose is to introduce non-linearity into the data, as real life data is likely to be non-linear. The convolution process is linear, so the ReLU function helps account for the non-linearity.

This image may help visualize what the ReLU step is doing:

![](https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-07-at-6-18-19-pm.png?w=748)

Other activation functions can be used besides ReLU, but ReLU has been found to typically perform the best. 


## Pooling Layer

The function of a pooling layer is to do dimensionality reduction on the convolution layer output. This helps reduce the amount of computation necessary, as well as prevent overfitting. It is common to insert a pooling layer after several convolutional layers. 

Two types of pooling layers are Max and Average. Max pooling will take the largest number in a defined spatial neighborhood. Average pooling will take the average value of the spatial neighborhood. Max pooling is the type of pooling typically used, as it has been found to perform better. 

There are also Global versions of both of these types of pooling. Global (average or max) pooling is a more extreme method of dimensionality reduction, it averages the input into one value per feature map. A Global pooling layer is often added towards the end of a model, right before the Dense output layer. 

Here's an example of how the Max and Sum (another type of pooling) layers look:

![](https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-07-at-6-11-53-pm.png?w=748)

In Keras pooling layers are added with the following functions:
```python
keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='valid')
keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
keras.layers.AveragePooling1D(pool_size=2, strides=None, padding='valid')
keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```

Global Pooling layers:
```python
keras.layers.GlobalMaxPooling1D()
keras.layers.GlobalAveragePooling1D()
keras.layers.GlobalMaxPooling2D(data_format=None)
keras.layers.GlobalAveragePooling2D(data_format=None)
```

The argument we care about is:
* pool_size - size of the window to 'summarize'

Note that the Global pooling layers do not have any input functions (except data_format for the 2D ones).


