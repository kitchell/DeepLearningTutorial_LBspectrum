# Loss Functions and Optimizers

When compiling your model you need to choose a loss function and an optimizer. The loss function is the quantity that will be minimized during training. The optimizer determines how the network will be updated based on the loss function. 

Example compile step:
```python
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
```

## Loss Functions

There are some simple guidlines for choosing the correct loss function:

**binary crossentropy** (`binary_crossentropy`) is used when you have a two-class, or binary, classification problem. 

**categorical crossentropy** (`categorical_crossentropy`) is uses for a multi-class classification problem. 

**mean squared error** (`mean_squared_error`) is used for a regression problem. 

In general, crossentropy loss functions are best to use when the model you use is outputting probabilities. 

Here is the [Keras documentaiton for loss functions](https://keras.io/losses/)


## Optimizers

There are many optimizers you can use. Here are the most common optimizers you might be interested in using:

### stochastic gradient descent

```python
keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
```

This is a common 'basic' optimizer and several other optimizers are variants of this. It can be adjusted by changing the learning rate, momentum and decay.

* learning rate (lr) - 
* momentum - accelerates SGD in the relevant direction and dampens oscillations
* decay - 



### RMSprop
```python
keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
```

Keras recommends that you only adjust the learning rate of this optimzer. 

### Adam
```python
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
```













The loss functions, metrics, and optimizers can be customized and configured like so:
```python
from keras import optimizers
from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
```

can optimize learning rate. 
