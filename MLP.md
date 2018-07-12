# Multi-Layer Perceptrons: Fully Connected Neural Networks

A multi-layer perceptron (MLP) is a fully connected neural network, meaning that each node connects to all possible nodes in the surrounding layers. The general format of the MLP has already been described in the last two pages. Here we will focus on how to create them using Keras. We will go through two examples given in the Keras documentation. These examples can be found [here](https://keras.io/getting-started/sequential-model-guide/).

### Dense Layer

To create a MLP or fully connected neural network in Keras, you will need to use the **Dense** layer. The Keras documentation on the Dense layer can be found [here](https://keras.io/layers/core/). A **Dense** layer is a fully connected layer. 

```python
keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```
The arguments we care about for the dense layer:
* units - Number of nodes in the hidden layer
* activation - activation function to use

Please see the Keras documentation for information on the others. 

### Dropout Layer

You may also need a **Dropout** layer. A **Dropout** layer helps prevent overfitting of the data. It randomly sets a fraction (defined by the user) of the input to 0 at each update during training. [Keras documenation on Dropout](https://keras.io/layers/core/#dropout) and [detailed information on Dropout](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf). Dropout layers are not required, however they are helpful. 

```python
keras.layers.Dropout(rate, noise_shape=None, seed=None)
```
The argument we care about for the dropout layer:
* rate - the fraction of the input units to drop

Please see the Keras documentation for information on the others.

## MLP for binary classification

Here is all of the code for a simple binary classification MLP example. We will go through it below.

```python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Generate dummy data
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))

model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
```

1. Import the libraries needed for the script
```python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
```
2. Create some data to use for the example
```python
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))
```
3. Define the type of model (Sequential)
```python
model = Sequential()
```
4. Add the first layer. Since this is an input layer we need to add an aditional argument (input_dim). This is a dense layer will 64 nodes. The data we are inputting has 20 values per input, so we tell it that the input dimension is 20. We also set the activation function to be 'relu'. 
```python
model.add(Dense(64, input_dim=20, activation='relu'))
```
5. Add a dropout layer to prevent overfitting. We have set the layer to drop 50% of the input.
```python
model.add(Dropout(0.5))
```
6. Add another hidden dense layer. This layer also has 64 nodes and uses the relu activation function. 
```python
model.add(Dense(64, activation='relu'))
```
7. Add another dropout layer, 50% rate again.
```python
model.add(Dropout(0.5))
```
8. Add the output layer. This is a dense layer with 1 node. It is one node because this is a binary classification problem, the output is either 1 or 0. The activation function used is sigmoid. Sigmoid is the most appropriate function to use for a binary classification problem because it forces the output to be between 0 and 1, making it easy to set a threshold (i.e. .5) for classification. 
```python
model.add(Dense(1, activation='sigmoid'))
```
9. Compile the model. Because this is a binary classification problem, we use the loss function 'binary_crossentropy'. The optimizer chosen is 'rmsprop' and we want it to output the accuracy metric. 
```python
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
```
10. Fit the model. This is what actually trains the model. We give it the input (training data) x_train and y_train. We ask it to run the training 20 times and use a batch size of 128. This means it will see 128 inputs before updating the weights. 
```python
model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
```
11. Last step is to check the accuracy of the model on some testing data that was kept out of training. 
```python
score = model.evaluate(x_test, y_test, batch_size=128)
```

## MLP for multi-class classification

Here is all of the code for a multi-class classification example. We will go through it below.
```python
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np

# Generate dummy data
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
```

1. Input the necessary libraries
```python
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
```
2. Create some data to use for the example.
```python
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
```
3. Define the type of model (Sequential) and add the hidden layers. These hidden layers are the exact same as above so I will not go through them one on one again.
```python
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
```
4. Add the output layer. Here is where it is different from the binary classification MLP. We have 10 possible classification categories, so we need 10 nodes for the output layer. We also use the softmax activation function. This is the best one to use for multi-class classification as softmax assigns decimal probabilities to each class in a multi-class problem. Those decimal probabilities must add up to 1.0 and this additional constraint helps training converge more quickly than it otherwise would.
```python
model.add(Dense(10, activation='softmax'))
```
5. Compile the model. In this example, we are using stochastic gradient descent (SGD,sgd) for the optimizer. The first line below allows us to cutomize agruments for the optimizer. We use the categorical_crossentropy loss function because it is a multi-class classification network. 
```python
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
```
6. Fit the model. We fit the model using the training data and 20 rounds of training.
```python
model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
```
7. Evaluate the model. Finally, we test the accuracy of the model using testing data we kept out of the training.
```python
score = model.evaluate(x_test, y_test, batch_size=128)
```

That's it! You now know the basic requirements of an MLP in Keras. Essentially, you need a dense input layer, some dense hidden layers (with or without dropout), and a dense output layer. 

Please continue on to [Convolutional Neural Networks](https://github.com/kitchell/DeepLearningTutorial_LBspectrum/blob/master/CNN.md).
