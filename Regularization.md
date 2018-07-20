# Regularization

We have already learning that overfitting is when the model becomes too specific and too well trained to the training data. One way this is avoided is by validating and testing the network on new data the model has not see before. There are a few additional ways we can prevent overfitting. One way is to modulate the amount of information the model model is allowed to store. If the model can only learned a small number of patterns, the optimization process will force it to focus on the most important patterns. This process is called **regularization**. 

The most common methods of regularization include:

### Reducing the network size

Reducing the number of layers and number of units/nodes per layer is a simple way to regularize your network. 

### Weight regularization

When training a network it is possible that multiple sets of weight values could explain the data. *Weight regularization* is based off of the idea that the 'simplest' model is likely the best one or the one least likely to overfit. A 'simple model' is a model where the distribution of parameter values has less entropy. A way to prevent overfitting is to put constraints on the complexity of a network and forcing its weights to only take small values, making the distribution more *regular*. The constraints are added by adding a cost for large weights to the loss function of the network. There are two cost options:

* L1 - the cost is proportional to the absolute value of the weight coefficients (also called the L1 norm of the weights
* L2 - the cost is proportional to the square of the value of the weight coefficients (also called the L2 norem of the weights and weight decay)

In Keras, weight regularization is added by passing weight regularizers instances to the `kernal_regularizer` argument when adding a layer to the model. [Keras documentation](https://keras.io/regularizers/).

**L1**
```python
keras.regularizers.l1(0.)
```

**L2**
```python
keras.regularizers.l2(0.)
```

**L1 and L2**
```python
keras.regularizers.l1_l2(0.)
```

The number in the parenthesis is what is multiplied to the weight coefficient value and added to the total loss of the network.

```python
from keras import regularizers

model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

### Dropout

Dropout has already been discussed but I will mention it again here. Dropout consists of 'dropping' a set percentage of training samples each training round. This is done by adding a dropout layer to the model

```python
model.add(layers.Dropout(0.5))
```

### Useful resources
http://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/4.4-overfitting-and-underfitting.ipynb


Please move on to [Hyperparameter Tuning](https://github.com/kitchell/DeepLearningTutorial_LBspectrum/blob/master/HyperparamTuning.md).
