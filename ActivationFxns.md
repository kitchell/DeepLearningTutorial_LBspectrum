# Activation Functions

Activation functions are what decide whether a neuron should be activated or not. It is the non linear transformation that is done over the input signal. This transformed output is then sent to the next layer of neurons as input.

* [useful resource](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0)
* [another useful resource](https://www.analyticsvidhya.com/blog/2017/10/fundamentals-deep-learning-activation-functions-when-to-use-them/)
* [another useful resource and source of the images used below](http://cs231n.github.io/neural-networks-1/#actfun)
* [yet another useful resource](http://cs231n.github.io/neural-networks-1/#actfun)

There are many possible input functions. See the [Keras documentation for a list of all of them](https://keras.io/activations/). The most commonly used functions are Sigmoid, Tanh, ReLU, and softmax. 

## Sigmoid

The sigmoid function 'squashes' the value so that it is between 0 and 1. It has the mathmatical formula: σ(x)=1/(1+e^(−x)). Large negative values will become 0, while large positive values will become 1. The sigmoid function is rarely used as the other functions typically do better. It is however often used with the output layer in binary classification. This is because it makes it easy to set a threshold (e.g. .5) for the two output categories. 

![](http://cs231n.github.io/assets/nn1/sigmoid.jpeg)

## Tanh

The tanh function is similar to the sigmoid function (actually it is just a scaled sigmoid) except it is 0 centered and 'squashes' the values so they are between -1 and 1. It has the mathmatical formula: tanh(x)=2σ(2x)−1. 

![](http://cs231n.github.io/assets/nn1/tanh.jpeg)

## ReLU

The rectified linear unit (ReLU) function is one of the most popular activation functions. Its matmatical formula is: f(x)=max(0,x). Essentially the ReLU function sets every negative value to 0, making the activation threshold simply 0. 

![](http://cs231n.github.io/assets/nn1/relu.jpeg)

## Softmax

The softmax function is useful when doing multi-class classification. It is typically only used with the output layer. It is a type of sigmoid function and it 'squashes' the outputs for each class between 0 and 1 and divides by the sum of the outputs. It essentially gives the probability of the input being in a particular class. Its mathematical function is: ![](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/10/17014509/softmax.png). 

Please continue on to [Other Useful Keras Functions](https://github.com/kitchell/DeepLearningTutorial_LBspectrum/blob/master/UsefulKerasFxns.md). 
