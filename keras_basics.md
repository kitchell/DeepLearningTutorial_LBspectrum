# Introduction to Keras

Keras is a high level, very powerful, and easy to use library for deep learning in Python. It runs on top of multiple other libraries and uses them as a backend, such as Tensorflow or Theano. I will be using a tensorflow backend, but it should run the same no matter what backend is used. 

### Useful Resources

https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-7
https://github.com/keras-team/keras/tree/master/examples
https://github.com/leriomaggio/deep-learning-keras-tensorflow/blob/euroscipy2017/1%20Multi-layer%20Fully%20Connected%20Networks.ipynb
http://neuralnetworksanddeeplearning.com/chap1.html
http://www.zhanjunlang.com/resources/tutorial/Deep%20Learning%20with%20Keras.pdf
https://www.youtube.com/channel/UC9OeZkIwhzfv-_Cb7fCikLQ

## Models in Keras

Models in Keras are defined as a sequence of layers. A Sequential model is created first and layers are added one at a time until you are happy with the topology. 

### example of defined, compiled, fit and tested model (after data is preprocessed and libraries imported

```
X = input_values #shape is 100 by 8
Y = input_categories #shape is 100 by 3 (one-hot encoding)

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu')
model.add(Dense(3, activation='softmax')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=150, batch_size=10)
scores = model.evaluate(X, Y)

```

### Defining the network structure

We are going to just us fake data for now. Focus on learning the general format, details will be discussed later. Lets say we have 100 subjects and 8 variables per subject and there are 3 possible categories. You will likely want to split the data up for cross validation, but for now we will ignore that as we are only interested in learning the parts of a Keras model. Lets build a simple fully connected network structure. I will go through the above code step by step

1. Define the data
```
X = input_values #shape is 100 by 8
Y = input_categories #shape is 100 by 3 (one-hot encoding)
```

2. Define the model as a sequential model
```
model = Sequential()
```
3. Add the input and first hidden layers. Here we use a fully connected layer which is defined by the 'Dense' class in Keras. We tell the model to create an input layer with 8 nodes by setting the `input_dim` variable to 8. The 12 tells it to create a Dense hidden layer with 12 nodes. The 'relu' tells the layer to use the 'relu' activation function (more on that later).
```
model.add(Dense(12, input_dim=8, activation='relu'))
```
4. Add another fully connected hidden layer. This time with 8 nodes, still with the relu activation function. Notice that we did not have to set the input dimension. You only set the input dimension for the first layer added. 

```
model.add(Dense(8, activation='relu')
```
5. Add the output layer. It has 3 nodes, one for each possible category. It uses the 'softmax' activation function. 
```
model.add(Dense(3, activation='softmax')
```
6. Compile the model. Now that the network is defined, we have to compile it. 
```
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

7. Fit the model. We need to train the model on the data.
```
model.fit(X, Y, epochs=150, batch_size=10)
```
8. Test the accuracy. Here we are testing the accuracy against the data it was trained on. It would be better to use cross validation, but this is a simple example. 
```
scores = model.evaluate(X, Y)
```

## Choices to be made when compiling
The compiling step prepares the model to be run by the backend. For this step we need to select a few options.

1. The **loss or cost function.** We need to select the function that is used by the optimizer to optimize the weights. [Here is a list of the available loss functions](https://keras.io/losses/). The most common are:
* mean_squared_error 
* binary_crossentropy - for binary label predictions
* categorical_crossentropy - for multi category label predictions

2. The **optimizer** algorithm that will be used to update the weights of the network. [Here is a list of the available optimizers](https://keras.io/optimizers/). The most common are:
* sgd - stochastic gradient descent
* adam - default 

3. The **metrics** we want to evaluate the model on. This is not used to train the model, but gets printed out as it trains. [Here is a list of the available metrics](https://keras.io/metrics/). The most common is:
* accuracy
