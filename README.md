# Deep Learning intro

Collection of completed exercises from the Udacity course  [Intro to Deep Learning with PyTorch](https://www.udacity.com/enrollment/ud188).  

The repository with the original exercises is in [deep-learning-v2-pytorch](https://github.com/udacity/deep-learning-v2-pytorch/). Iconsists of a bunch of tutorial notebooks for various deep learning topics.

# Table of contents

The topics covered in the exercises are

## Introduction to Neural Networks

 - **Introduction and implementation of gradient descent**: 
	 - To help with the visualisation of the data, I wrote a Flask application to draw lines in with the parametric equation, [ApplicationForLines](https://github.com/MartinezAgullo/DeepLearning_intro/tree/main/ApplicationForLines). 
	 - The exercises of this part are on [01_PerceptronAlgorithm](https://github.com/MartinezAgullo/DeepLearning_intro/tree/main/01_PerceptronAlgorithm), [02_Softmax_and_CrossEntropy](https://github.com/MartinezAgullo/DeepLearning_intro/tree/main/02_Softmax_and_CrossEntropy), [03_GradientDescent](https://github.com/MartinezAgullo/DeepLearning_intro/tree/main/03_GradientDescent), [04_AnalyzingStudentData](https://github.com/MartinezAgullo/DeepLearning_intro/tree/main/04_AnalyzingStudentData).
 - **Introduction to PyTorch**: 
	 - In [05_Tensors](https://github.com/MartinezAgullo/DeepLearning_intro/tree/main/05_Tensors) the basic objects of PyTorch are defined.
	-  In [06_NN_MNIST](https://github.com/MartinezAgullo/DeepLearning_intro/tree/main/06_NN_MNIST) a NNs is defined and trained to classify the MNIST dataset.
	- In [07_ClassifyingClothes](https://github.com/MartinezAgullo/DeepLearning_intro/tree/main/07_ClassifyingClothes), we used a trained NN for the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) dataset is used. This is a more complex dataset than the MNIST.
	 -  We save and load trained models in [08_LoadingAndSavingModels](https://github.com/MartinezAgullo/DeepLearning_intro/tree/main/08_LoadingAndSavingModels).
	 - Use transfer learning to train a state-of-the-art image classifier for dogs and cats in [09_TransferLearning](https://github.com/MartinezAgullo/DeepLearning_intro/tree/main/09_TransferLearning).


## Convolutional Neural Networks
- **CNN**: In [10_ConvVisualization](https://github.com/MartinezAgullo/DeepLearning_intro/tree/main/10_ConvVisualization) we
	- Visualise the output of layers that make up a CNN.
	- Define and train a CNN for classifying [MNIST data](https://en.wikipedia.org/wiki/MNIST_database).
	- Define and train a CNN for classifying images in the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
- **Style transfer**: Extract style and content features from images, using a pre-trained network. See [12_StyleTransfer](https://github.com/MartinezAgullo/DeepLearning_intro/tree/main/12_StyleTransfer).

## Recurrent Neural Networks  
- In extracting information about the sequence of data with RNNs in [13_RNN_and_LSTM](https://github.com/MartinezAgullo/DeepLearning_intro/tree/main/13_RNN_and_LSTM):
	- Simple example in [time-series](https://github.com/MartinezAgullo/DeepLearning_intro/tree/main/13_RNN_and_LSTM/time-series).
	- Character prediction in [char-rnn](https://github.com/MartinezAgullo/DeepLearning_intro/tree/main/13_RNN_and_LSTM/char-rnn).
- Implementation of a recurrent neural network with LSTM (Long Short-Term Memory) that can predict if the text of a movie review is positive or negative in [14_SentimentAnalysis](https://github.com/MartinezAgullo/DeepLearning_intro/tree/main/14_SentimentAnalysis).

