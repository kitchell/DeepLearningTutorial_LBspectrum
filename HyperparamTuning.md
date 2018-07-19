# Hyperparameter Tuning

A way to improve your model and avoid overfitting is to tune the hyperparameters. Hyperparameters are the parameters that are not learned within the model. These are the choices you have to make when defining the model, like how many hidden nodes to use?, how much dropout to add?, which activation function? etc. There are no set rules for choosing many of these hyperparameters, so it may be beneficial for you to test multiple combinations of each parameter. This can be done in many ways, such as through a grid search or random search. 

## Grid Search

A grid search exhaustively tests all combinations of a grid of parameters selected. This can easily be implemented using the `GridSearchCV` function from sklearn. [sklearn documentation](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV).

```python
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

#create a function for defining the network
def make_model(dense_layer_sizes, filters, kernel_size, pool_size):

    model = Sequential()

    model.add(Conv2D(filters, (kernel_size, kernel_size),
                     padding='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(filters, (kernel_size, kernel_size)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    for layer_size in dense_layer_sizes:
        model.add(Dense(layer_size))
        model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    return model 

dense_size_candidates = [[32], [64], [32, 32], [64, 64]]

#create a classifier using the model function
my_classifier = KerasClassifier(make_model, batch_size=32)

#define the grid search 
validator = GridSearchCV(my_classifier,
                         param_grid={'dense_layer_sizes': dense_size_candidates,
                                     # nb_epoch is avail for tuning even when not
                                     # an argument to model building function
                                     'epochs': [3, 6],
                                     'filters': [8],
                                     'kernel_size': [3],
                                     'pool_size': [2]},
                         scoring='neg_log_loss',
                         n_jobs=1)
validator.fit(X_train, y_train)

print('The parameters of the best model are: ')
print(validator.best_params_)

# validator.best_estimator_ returns sklearn-wrapped version of best model.
# validator.best_estimator_.model returns the (unwrapped) keras model
best_model = validator.best_estimator_.model
metric_names = best_model.metrics_names
metric_values = best_model.evaluate(X_test, y_test)
for metric, value in zip(metric_names, metric_values):
    print(metric, ': ', value)
```

## Random Search

Sklearn also has a function for performing a random search of hyperparameter values, `RandomizedSearchCV`. Instead of trying all parameters it randomly selects the paramters a set number of times.  [sklearn documentation](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV). 

The set up is essentially the same as the grid search, except you also have to set a number of iterations. 

```python
validator = RandomSearchCV(my_classifier,
                         param_grid={'dense_layer_sizes': dense_size_candidates,
                                     # nb_epoch is avail for tuning even when not
                                     # an argument to model building function
                                     'epochs': [3, 6],
                                     'filters': [8],
                                     'kernel_size': [3],
                                     'pool_size': [2]},
                         n_iter=20,
                         scoring='neg_log_loss',
                         n_jobs=1)
```

[Here's an example comparing the two types of search in sklearn](http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py).

## Hyperas

There is a package called [Hyperas](http://maxpumperla.com/hyperas/) that combines another package called [Hyperopt](http://hyperopt.github.io/hyperopt/) and Keras for fast hyperparameter optimization.


### useful resources
http://neupy.com/2016/12/17/hyperparameter_optimization_for_neural_networks.html

https://github.com/leriomaggio/deep-learning-keras-tensorflow/blob/pydata-london2017/4.%20HyperParameter%20Tuning.ipynb

http://scikit-learn.org/stable/modules/grid_search.html
