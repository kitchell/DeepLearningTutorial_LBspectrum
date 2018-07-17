# Data Preprocessing

Before feeding your data into a neural network there is some data preparation that must be done. The exact preprocessing that must be done will be different for each domain but we can cover the basics here. 

## Vectorization

All inputs and targets (data and labels) fed into a neural network must be tensors of floating point data. The step of turning the data into a tensor is called *vectorization*. A tensor is ...

examples...

For the data labels this is done using one-hot encoding, as we have seen already. 


## Normalization

The input data should be normalized so that each feature has a mean of 0 and a standard deviation of 1. It is not good to feed in large values or heterogeneous data (e.g. data where one feature is in the 0-1 range and another is in the 200-300 range), as it will cause issues with the gradient descent. Instead, the data should have small values (most in the 0-1 range) and be homogeneous (all features have data in roughly the same range). 

The normalization step is done by normalizing each feature independently to have a mean of 0 and noramlizing each feature independently to have a standard deviation of 1. This can be done manually or by using a function in sklearn.

### Manually
```python
x = data #numpy array

# subtract the mean across the 0-axis (columns)
x -= x.mean(axis=0)

# divide by the standard deviation across the 0-axis (columns(
x /= x.std(axis=0)
```

### sklearn
sklearn has a function called StandardScaler that will do this for you. [sklearn documentation](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

```python
X = data 

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
```

In the case of the Laplace Beltrami data, the spectrum is already vectorized. All we will need to do is one-hot encode the labels and normalize the eigenvalues. 



