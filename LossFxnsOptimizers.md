# Loss Functions and Optimizers

When compiling your model you need to choose a loss function and an optimizer. The loss function is the quantity that will be minimized during training. The optimizer determines how the network will be updated based on the loss function. 

Example compile step:
```python
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
```

## Loss Functions

There are some simple guidlines for choosing the correct loss function:

**binary crossentropy** is used when you have a two-class, or binary, classification problem. 

**categorical crossentropy** is uses for a multi-class classification problem. 

**mean squared error** is used for a regression problem. 

**connectionist temporal classification** is used for a sequence learning problem. 

In general, crossentropy loss functions are best to use when the model you use is outputting probabilities. 



## Optimizers











The loss functions, metrics, and optimizers can be customized and configured like so:
```python
from keras import optimizers
from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
```

can optimize learning rate. 
