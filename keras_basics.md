# Introduction to Keras

Keras is a high level, very powerful, and easy to use library for deep learning in Python. It runs on top of multiple other libraries and uses them as a backend, such as Tensorflow or Theano. I will be using a Tensorflow backend, but it should run the same no matter what backend is used. 

### Useful Resources

[Keras Sequential model guide](https://keras.io/getting-started/sequential-model-guide/)

## Models in Keras

Models (or networks) in Keras are defined as a sequence of layers. A Sequential model is created first and layers are added one at a time until you are happy with the network topology. 

### Example of defined, compiled, fit and tested model (after data is preprocessed and libraries imported)

```python
X = test_samples #shape is 100 rows by 8 columns
Y = test_labels #shape is 100 by 3 (one-hot encoding)
X_val = validation_samples
Y_val = validation_labels

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu')
model.add(Dense(3, activation='softmax')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=150, batch_size=10)
scores = model.evaluate(X, Y)

```

### Defining the network structure

Lets say we have 100 samples and 8 variables per sample and there are 3 possible categories to which the samples belong. We also have a small set of samples set aside for validation of the model. Lets build a simple fully connected network structure. I will go through the above code step by step

1. Define the data. X and Y are the training data. They will be used to train the model. X_val and Y_val, are samples from the same dataset that were set aside for testing the model's accuracy. They are not used in the training process (more on how and why to do this later). 

```python
X = test_samples #shape is 100 rows by 8 columns
Y = test_labels #shape is 100 by 3 (one-hot encoding)
X_val = validation_samples
Y_val = validation_labels
```

2. Define the model as a sequential model
```python
model = Sequential()
```
3. Add the input and first hidden layers. Here we use a fully connected layer which is defined by the 'Dense' class in Keras. We tell the model to create an input layer with 8 nodes by setting the `input_dim` variable to 8. The 12 tells it to create a Dense hidden layer with 12 nodes. The 'relu' tells the layer to use the 'relu' activation function (more on that later).
```python
model.add(Dense(12, input_dim=8, activation='relu'))
```

4. Add another fully connected hidden layer. This time with 8 nodes, still with the relu activation function. Notice that we did not have to set the input dimension. You only set the input dimension for the first layer added. 

```python
model.add(Dense(8, activation='relu')
```

5. Add the output layer. It has 3 nodes, one for each possible category. It uses the 'softmax' activation function (recommended for multi-class classification). 
```python
model.add(Dense(3, activation='softmax')
```

6. Compile the model. Now that the network is defined, we have to compile it. This translates the model from Keras into the specific backend being used (Tensorflow in my case). 
```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

7. Fit the model. The step is where the model is actually trained on the data. 
```python
model.fit(X, Y, epochs=150, batch_size=10)
```

8. Test the accuracy. Here we are testing the accuracy with the validatiion samples we kept out of the training dataset.   
```python
scores = model.evaluate(X_val, Y_val)
```

## Choices to be made when adding layers
1. There are many choices to be made when adding layers, starting with the type of layer to add. You can find out more about the existing layers [here](https://keras.io/layers/about-keras-layers/) and you will learn more about the different layers and their options later. The main layers we are interested in are as follows:

* **Dense** layer - a dense layer is a fully connected neural network layer
* **Conv1D** layer - a 1 dimensional convolutional layer 
* **Conv2D** layer - a 2 dimensional convolutional layer
* **MaxPooling1D** layer - a 1 dimensional pooling layer
* **MaxPooling2D** layer - a 2 dimensional pooling layer
* **Dropout** layer - Dropout consists of randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.

2. The **activation function**. The activation function is what decides whether or not a neuron/node should be activated. [Here are the available activation functions](https://keras.io/activations/) The most popular ones are:
* **linear** - default
* **sigmoid** - best with binary classification
* **softmax** - best with multi-class classification
* **Tanh** - hyperbolic tangent
* **relu** - most popular

## Choices to be made when compiling
The compiling step prepares the model to be run by the backend. For this step we need to select a few options. [Keras documentation on compile](https://keras.io/models/model/)

1. The **loss or cost function.** We need to select the function that is used by the optimizer to optimize the weights. The loss function is how the network measures its performance. [Here is a list of the available loss functions](https://keras.io/losses/). The most common are:
* mean_squared_error - for regression
* binary_crossentropy - for binary label predictions
* categorical_crossentropy - for multi category label predictions

2. The **optimizer** algorithm that will be used to update the weights of the network. [Here is a list of the available optimizers](https://keras.io/optimizers/). The most common are:
* sgd - stochastic gradient descent, a fast variant of gradient descent
* RMSprop - more advanced
* adam - default, more advanced, includes a concept of momentum, most popular

3. The **metrics** we want to evaluate the model on. This is not used to train the model, but gets printed out as it trains. [Here is a list of the available metrics](https://keras.io/metrics/). The most common is:
* accuracy

## Choices to be made when fitting the model
The fitting step trains the model on the input data. For this step we need to select a few options. [Keras documentation on fit](https://keras.io/models/model/)

1. **epochs** - this is the number of times the model is exposed to the training set. At each iteration the optimizer tries to adjust the weights so that the loss function is minimized

2. **batch_size** - This is the number of training instances observed before the optimizer performs a weight update. 

## Other important information

### Input data

The input data (both samples and the labels) needs to be in the datatype format of 'float32'. This can be set using the `.astype()` function from numpy. 
```python
train_images = train_images.astype('float32')
```

The input labels can be converted from a simple list to one-hot encoding using the `to_categorical` function from keras. The labels should be in the format of a numpy array. 
```python
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
```

When setting up the shape of your input data, the first axis is generally the 'samples' axis. It should be equal to the number of samples in your data. The other values represent the shape of the samples. Example of a data set with 60,000 samples and each sample is a 28 by 28 matrix. 
```python
print(train_images.shape)
(60000, 28, 28)
```

You may need to rearrange your input data into a new shape. This can be done with the `reshape()` function in numpy. This is applicable, for example, when using fully connected layers with image data. You may need to convert the image from a matrix to a vector.
```python
train_images = train_images.reshape((60000, 28 * 28))
```
Please continue on to [Multi-layer Perceptrons in Keras](https://github.com/kitchell/DeepLearningTutorial_LBspectrum/blob/master/MLP.md).
