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

This prints out a vizual description of the model.

![](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/09/Plot-of-Neural-Network-Model-Graph.png)


## Callback functions

1. EarlyStopping 
choose a metric to monitor (e.g. val_loss) and stop training if that value has not changed for a certain number of epochs (e.g. 2).

2. ModelCheckpoint
stores the best model found during the different epoch.
