# Reimplement Neural Networks: Zero to Hero
I watch the playlist on YouTube [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy) - Neural Networks: Zero to hero. Then I recreate the code from the lectures from scratch and do exercises for the lectures. A course on neural networks that starts all the way at the basics. The Jupyter notebooks I build are then captured here inside the [lectures](https://github.com/olegdavydovai/reimplement-nn-zero-to-hero/tree/main/lectures) directory. More details about each completed project are written below ↓↓↓. [Here](https://github.com/karpathy/nn-zero-to-hero?tab=readme-ov-file) is the original GitHub of the course.

## Lecture 0: Autograd, basics of neural networks, backpropogation
Reimplement: I am building a Value class without libraries, from scratch in Python. It accepts data, builds an object for them and tracks the operations that go through the entire code. It knows how any value was obtained, as a result of which operations, and stores this information. Then it builds a topographic graph that tells how the code was executed from beginning to end. Due to this, back propagation is performed using chain rule differentiation. Now we know the gradients of each parameter and I update them so that the loss function decreases according to the maximum likelihood estimation rules. The loss decreases using vanilla gradient descent. I am implementing a Neuron class that accepts inputs, initializes the weights for the inputs and biases. A Layer class that consists of several independent neurons that accept the same inputs, but different weights and biases for them. An MLP class that consists of several consecutive layers.  

Exercises:
- [Jupyter file 0](https://github.com/olegdavydovai/reimplement-nn-zero-to-hero/blob/main/lectures/lecture_0_autograd(micrograd).ipynb)

## Lecture 1: Linear Trigram

Reimplement: I'm implement a trigram character-level language model, i.e. a neural network model take two characters as an input to predict the 3rd one. Training model, evaluation and sample new names from scratch. The dataset consists of people's names. I first rewind it randomly, and then  it: 1) 80% on a train set to train the neural network 2) 10% validation set to configure hyperparameters 3) 10% test set to check performance at the end on data that the neural network has not seen. Loss - the negative log likelihood for classification (cross_enropy). Regularization.

Exercises:
- [Jupyter file 1](https://github.com/olegdavydovai/reimplement-nn-zero-to-hero/blob/main/lectures/lecture_1_trigram_bigram.ipynb)

## Lecture 2: MLP N-gram

Reimplement: I continue to improve the code from the previous lecture. I make a multilayer perceptron (MLP) character-level language model. I make the number of characters for a context be chosen in one place. I split the data into training, validation, and test. Learning rate and hyperparameters tuning, under/overfitting, size of network for some number of data. Lookup table of embeddings, minibatch, stochastic gradient descent, size of embeddings, efficient concatenation of embeddings. Learning rate decay. Linear projection S [from an paper by Bengio et al](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) which looks like a residual(skip) connection, but is not. Tracking logs, what degree of error we should expect, how to initialize logits so that the loss is reduced during initialization. A sample of the trained model. The structure of embeddings from the lookup table (it is definitely not random). MY TEST LOSS = 2.1304

Exercises:
- [Jupyter file 2](https://github.com/olegdavydovai/reimplement-nn-zero-to-hero/blob/main/lectures/lecture_2_MLP_fourgram.ipynb)

## Lecture 3: Activations & Gradients

Reimplement: Set initial logits to approximately zero for good initial loss, Kaiming initialization. Variance, standart diviation and histograms of activations, running mean and std, gain and bias of batchnorm as parameters of network. Tracking and preventing dead neurons that do not receive gradients. Supporting neuron activations to have a standard deviation, tracking gradients so that they do not get small or large(vanishing and exploads gradients), so that the network can learn, simularity break(initialize weights to zero).

Exercises:
