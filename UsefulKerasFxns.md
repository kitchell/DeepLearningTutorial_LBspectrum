# Additional Useful Functions in Keras

model.summary()


## Callback functions

1. EarlyStopping 
choose a metric to monitor (e.g. val_loss) and stop training if that value has not changed for a certain number of epochs (e.g. 2).

2. ModelCheckpoint
stores the best model found during the different epoch.
