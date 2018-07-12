# Introduction to Keras

Keras is a high level, very powerful, and easy to use library for deep learning in Python. It runs on top of multiple other libraries and uses them as a backend, such as Tensorflow or Theano. I will be using a tensorflow backend, but it should run the same no matter what backend is used. 

### Useful Resources

[Keras Sequential model guide](https://keras.io/getting-started/sequential-model-guide/)

## Models in Keras

Models in Keras are defined as a sequence of layers. A Sequential model is created first and layers are added one at a time until you are happy with the topology. 

### Example of defined, compiled, fit and tested model (after data is preprocessed and libraries imported)

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

Lets say we have 100 subjects and 8 variables per subject and there are 3 possible categories to which the subjects belong. Lets build a simple fully connected network structure. I will go through the above code step by step

1. Define the data (made up for now. Typically, you will likely want to split the data up for cross validation, but for now we will ignore that as we are only interested in learning the general parts of a Keras model.)

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

## Choices to be made when adding layers (models)
1. There are many choices to be made when adding layers, starting with the type of layer to add. You can find out more about the existing layers [here](https://keras.io/layers/about-keras-layers/). The layers we will be interested in are as follows:

* **Dense** layer - a dense layer is a fully connected neural network layer
* **Conv1D** layer - a 1 dimensional convolutional layer 
* **Conv2D** layer - a 2 dimensional convolutional layer
* **MaxPooling1D** layer - a 1 dimensional pooling layer
* **MaxPooling2D** layer - a 2 dimensional pooling layer
* **Dropout** layer - Dropout consists of randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.

2. The **activation function**. The activation function is what decides whether or not a neuron/node should be activated. [Here are the available activation functions](https://keras.io/activations/) The most popular ones are:
* **linear**
* **sigmoid**
* **softmax**
* **Tanh** - hyperbolic tangent
* **relu** 

## Choices to be made when compiling
The compiling step prepares the model to be run by the backend. For this step we need to select a few options. [Keras documentation on compile](https://keras.io/models/model/)

1. The **loss or cost function.** We need to select the function that is used by the optimizer to optimize the weights. [Here is a list of the available loss functions](https://keras.io/losses/). The most common are:
* mean_squared_error 
* binary_crossentropy - for binary label predictions
* categorical_crossentropy - for multi category label predictions

2. The **optimizer** algorithm that will be used to update the weights of the network. [Here is a list of the available optimizers](https://keras.io/optimizers/). The most common are:
* sgd - stochastic gradient descent, a fast variant of gradient descent
* RMSprop - more advanced, includes a concept of momentum
* adam - default, more advanced, includes a concept of momentum

3. The **metrics** we want to evaluate the model on. This is not used to train the model, but gets printed out as it trains. [Here is a list of the available metrics](https://keras.io/metrics/). The most common is:
* accuracy

## Choices to be made when fitting the model
The fitting step trains the model on the input data. For this step we need to select a few options. [Keras documentation on fit](https://keras.io/models/model/)

1. **epochs** - this is the number of times the model is exposed to the training set. At each iteration the optimizer tries to adjust the weights so that the loss function is minimized

2. **batch_size** - This is the number of training instances observed before the optimizer performs a weght update. 

Please continue on to [Multi-layer Perceptrons in Keras](https://github.com/kitchell/DeepLearningTutorial_LBspectrum/blob/master/MLP.md).
