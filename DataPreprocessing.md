# Data Preprocessing

Before feeding your data into a neural network there is some data preparation that must be done. The exact preprocessing that must be done will be different for each domain but we can cover the basics here. 

## Vectorization

All inputs and targets (data and labels) fed into a neural network must be tensors of floating point data. The step of turning the data into a tensor is called *vectorization*. A tensor is a multidimensional array, a generalization of matrices to an arbitrary number of dimensions. For example, a matrix is a 2D tensor, a vector is a 1D tensor, a scalar is a 0D tensor etc. In python, these are represented using numpy arrays. Your input data should be in a numpy array with the data format 'float32'. Data labels can be vectorized using one-hot encoding, as we have seen already. 

Example of data tensors:

### Vector data

Vector data should be combined into a 2D array with the size (samples, features). One row per sample, one column per feature.

### Time series or Sequence data

Time series or sequence data should be in a 3D array. Each sample can be encoded as a 2D matrix (one row per feature, one column per timestep) and the samples are combined into a 3D array of size (samples, features, timesteps). 

### Image data

Image data should be in a 4D array of size (samples, image height, image width, number of channels). The number of channels referes to the color channels of the image. If it is a greyscale image, then the number of channels is 1. If it is a color image, then the number of channels is 3 (RGB or HSV). 


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


Please continue on to [Regularization](https://github.com/kitchell/DeepLearningTutorial_LBspectrum/blob/master/Regularization.md).

