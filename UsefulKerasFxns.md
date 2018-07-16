# Additional Useful Functions in Keras


## Information about the model


```python
model.summary() 
```
This provides a print out summarizing the network like this:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_2 (Dense)              (None, 100)               9400      
_________________________________________________________________
dense_3 (Dense)              (None, 9)                 909       
_________________________________________________________________
activation_2 (Activation)    (None, 9)                 0         
=================================================================
Total params: 10,309
Trainable params: 10,309
Non-trainable params: 0
_________________________________________________________________
```

```python
from keras.utils import plot_model
plot_model(model, to_file='model.png')
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
```

This prints out a visual description of the model.

![](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/09/Plot-of-Neural-Network-Model-Graph.png)

You can also iterate through the layers and print information:
```python
print('Model Input Tensors: ', model.input, end='\n\n')
print('Layers - Network Configuration:', end='\n\n')
for layer in model.layers:
    print(layer.name, layer.trainable)
    print('Layer Configuration:')
    print(layer.get_config(), end='\n{}\n'.format('----'*10))
print('Model Output Tensors: ', model.output)
```

## Callback functions

Callback functions are included in the fitting step of model creation:
```python
model.fit(X_train, Y_train, validation_data = (X_val, Y_val), epochs=50, 
          batch_size=128, verbose=True, callbacks=[best_model, early_stop])
```

[Keras Documentation on Callback functions](https://keras.io/callbacks/)

### Model History
```python
keras.callbacks.History()

# list all data in history
history = model.fit(...)
print(history.history.keys())
```
By default, the fit function returns the entire history of training/validation loss and accuracy, for each epoch. We can therefore plot the behaviour of loss and accuracy during the training phase:
```python
import matplotlib.pyplot as plt
%matplotlib inline

def plot_history(network_history):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(network_history.history['loss'])
    plt.plot(network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(network_history.history['acc'])
    plt.plot(network_history.history['val_acc'])
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()

plot_history(network_history)
```

### Early Stopping

```python
keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None)

early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1) 
```
The early stopping callback stops training once a monitored metric stops improving. Monitor is set to the quanity to monitor (e.g. val_loss) and patience is set to the number of epochs with no change (e.g. 2).

### Model Checkpoint

```python
keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

#save the best model
best_model = ModelCheckpoint(fBestModel, verbose=0, save_best_only=True)
best_model = ModelCheckpoint(fBestModel, monitor='val_acc', save_best_only=True, mode='max')
```
The model checkpoint callback stores the model after every epoch. It can also be set to save the the best model found during the epochs. 

### TensorBoard
```python
keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)
```
The tensorboard callback writes a log for use with the [TensorBoard visualization](https://www.tensorflow.org/guide/summaries_and_tensorboard).


### CSV logger
```python
keras.callbacks.CSVLogger(filename, separator=',', append=False)
```
The CSV logger streams the epoch results to a csv file. 

## Other functions

### Batch normalization

When we initialize a model, we typically normalize the inital values of our input to have 0 mean and unit variance. As training progresses we may loose this normalization, slowing training and causing issues. A batch normalization layer reestablishes these normalizations. [Keras documentation](https://keras.io/layers/normalization/).

```python
from keras.layers.normalization import BatchNormalization

BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, 
                   beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                   moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                   beta_constraint=None, gamma_constraint=None)
```

## Tips

You can use a validation set of data during training to see how well the model is generalizing. [Better explanation here](https://stackoverflow.com/questions/46308374/what-is-validation-data-used-for-in-a-keras-sequential-model). This data will not be used to train the model, but it gives you an idea of how well it does without having to fully evaluate the model with the test data. 
```python
# Train model (use 10% of training set as validation set)
history = model.fit(X_train, Y_train, validation_split=0.1)

# Train model (use validation data as validation set)
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val))
```

[View the hidden representations](https://github.com/leriomaggio/deep-learning-keras-tensorflow/blob/pydata-london2017/2.1%20Hidden%20Layer%20Representation%20and%20Embeddings.ipynb) (scroll down to end)

Please continue on to [Loss functions and optimizers](https://github.com/kitchell/DeepLearningTutorial_LBspectrum/blob/master/LossFxnsOptimizers.md).
