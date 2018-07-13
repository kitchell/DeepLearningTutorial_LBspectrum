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


## Callback functions

Callback functions are included in the fitting step of model creation:
```python
model.fit(X_train, Y_train, validation_data = (X_val, Y_val), epochs=50, 
          batch_size=128, verbose=True, callbacks=[best_model, early_stop])
```

[Keras Documentation on Callback functions](https://keras.io/callbacks/)

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
