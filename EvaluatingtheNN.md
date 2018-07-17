# Evaluating the Neural Network

A main goal of machine learning is to create models that *generalize*, meaning that they perform well on data that has not been seen before. To test the generalization power of a model you typically need to split your available data into three separate datasets: **a training set, a validation set, and a testing set**. You then train the model on the training data and evaluate its performance with the validation data. Once the model is sufficiently accurate, you test it a final time on the testing data. It is best to test it on a completely new third (testing) dataset to avoid overfitting to the validation data. When you are training the model, you are making descision based on how accurately it classifies (or predicts etc.) the validation data. This inadvertently teaches the model what the validation data looks like and it is possible to overfit and overoptimize it for the validation data even if the model never sees the data in training.

First, some terms:

![](https://cdn-images-1.medium.com/max/1600/1*tBErXYVvTw2jSUYK7thU2A.png)

### Underfitting

Underfitting is when the model does not fit the training data very well and therefore cannot generalized to the validation or testing data. The model has poor predictive abilities and its accuracy is low. 

### Overfitting

Overfitting is when the model fits the training or validation data too well. It becomes too specialized for that data and cannot generalize to new data. It has very high accuracy for the training data, but performs poorly when evaluated with new data. 


There are a couple of ways to split the data to help avoid these problems: 

## Hold out validation

Hold out validation is simply splitting the data into multiple groups and 'holding' one or more groups (testing/validation) 'out' from training. This works ok, but sometime the split is not as random as we would like. 

### Manually 
This is example, semi-pseudocode. You would also need to do the same with the labels.

```python
#set the number of samples you want to use as validation data
num_validation_samples = 10000

#set the number of samples you want to use as testing data
num_test_samples = 5000

#shuffle the data
np.random.shuffle(data)

#extract the testing data (select the first 5000 samples)
test_data = data[:num_test_samples]

#extract the validation data (select the sample entries between 5000 and 15000)
validation_data = data[num_test_samples:num_validation_samples+num_test_samples]

#extract the training data (select the sample entries from 15000 to the end)
training_data = data[num_validation_samples+num_test_samples:]

# At this point you can tune your model,
# retrain it, evaluate it, tune it again...
model = get_model()
model.train(training_data)
validation_score = model.evaluate(validation_data)

# Once model is sufficiently tuned, evaluate it on the test data
model = get_model()
model.train(np.concatenate([training_data,
validation_data]))
test_score = model.evaluate(test_data)
```

### Using sklearn
This is probably the easiest option. sklearn has a built in function for splitting up datasets and the labels as well, however it will only split the data into two sets (a training and testing set). You can simply use the function twice. 

```python
from sklearn.model_selection import train_test_split

#set up the data
X = data
Y = data_labels

#choose what proportion of the data you want to be used for training and validation
test_size = .8

#split the data into 80% training (train + val) and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

test_size = .75
#you can use the function again to split up the training/validation data
X_train, X_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=test_size)

#this splits the data:
- 20% test
- 60% train
- 20% validation
```

## Cross Validation

### Useful resources:
https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6
