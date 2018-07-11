# Tutorial for applying 'Deep Learning' neural networks to the Laplace Beltrami shape data

In this tutorial we will show how to implement several different types of neural networks with the shape descriptor computed by the Laplace Beltrami operator.

First we need to cover some basic information and terms.

## Deep learning vs. Machine learning

### Machine learning

Machine learning is a subfield of artificial intelligence that focuses on teaching computers how to learn without being specifically programmed for certain tasks. The idea is to create algorithms that learn from data and make predictions on data. Machine learning can be split into three categories:
1. <ins>Supervised learning</ins>: the computer is given training data and the desired output from that data (e.g. category labels) and it has to learn from the training data in order to make meaningful predictions based on new data.

2. <ins>Unsupervised learning</ins>: the computer is given data only and it has to find meaningful structure or groups in the data by itself with no supervision. 

3. <ins>Reinforcement learning</ins>: the computer is interacting with the environment and learning via feedback which behaviors generate rewards. 


### Deep learning

Deep learning is a subset of machine learning methods that use artifical neural networks. It is somewhat inspired by the way the structure of the neurons of the brain. The "Deep" in deep learning refers to the multiple hidden layers of the neural networks. Deep learning has had great success with several domains, such as images, text, speech, and video. There are many differen types of neural networks. We will focus on the following:
1. 
