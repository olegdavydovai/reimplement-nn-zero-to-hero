# Reimplement Neural Networks: Zero to Hero
I watch the playlist on YouTube [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy) - Neural Networks: Zero to hero. Then I recreate the code from the lectures from scratch and do exercises for the lectures. A course on neural networks that starts all the way at the basics. The Jupyter notebooks I build are then captured here inside the [lectures](https://github.com/olegdavydovai/reimplement-nn-zero-to-hero/tree/main/lectures) directory. More details about each completed project are written below ↓↓↓. [Here](https://github.com/karpathy/nn-zero-to-hero?tab=readme-ov-file) is the original GitHub of the course.

## Table of Contents
- [Lecture 0: Autograd, basics of neural networks, backpropagation](#lecture-0-autograd-basics-of-neural-networks-backpropogation)
- [Lecture 1: Linear Trigram](#lecture-1-linear-trigram)
- [Lecture 2: MLP N-gram](#lecture-2-mlp-n-gram)
- [Lecture 3: Activations & Gradients](#lecture-3-activations--gradients)

## Lecture 0: Autograd, basics of neural networks, backpropogation
**Reimplement and study**: Derivative is core of backpropagation. I am building a Value, Neuron, Layer, MLP classes without libraries, from scratch in Python. Value accepts data, builds an object for them and tracks the operations that go through the entire code. It knows how any value was obtained, as a result of which operations, and stores this information. Then it builds a topographic graph. In this regard, backpropagation is performed using chain rule differentiation and the gradient of each parameter with respect to the loss function is obtained. I update them so that the loss function decreases according to the maximum likelihood estimation rules. Optimization is done using vanilla gradient descent. I am implementing a Neuron class that accepts inputs, initializes the weights and biases for the inputs. A Layer class that consists of several independent neurons that accept the same inputs, but different weights and biases for them. An MLP class that consists of several consecutive layers. Don't forget about zero grad. The nonlinearity I use is tanh, loss is MSE.

**Exercises**: Differentiation of an equation through a set of formulas and through a single one. Define log() and the rest that was included in the Value class. Implementation of logits.grad with pytorch.
- [Jupyter file 0](https://github.com/olegdavydovai/reimplement-nn-zero-to-hero/blob/main/lectures/lecture_0_autograd(micrograd).ipynb)

## Lecture 1: Linear Trigram
**Reimplement and study**: I'm implement a trigram character-level language model, i.e. a neural network model take two characters as an input to predict the 3rd one. The dataset consists of people's names. I training model, evaluation it and sample new names from scratch. Loss - the negative log likelihood for classification (cross_entropy). Regularization.

**Exercises**: Upgrade Bigram to Trigram. Split up the dataset randomly into 80% train set, 10% dev set, 10% test set. Train the bigram and trigram models only on the training set. Evaluate them on dev and test splits. Use the dev set to tune the strength of regularization. Instead of using one_hot pull the row from Weights, use F.cross_entropy with pytorch.
- [Jupyter file 1](https://github.com/olegdavydovai/reimplement-nn-zero-to-hero/blob/main/lectures/lecture_1_trigram_bigram.ipynb)

## Lecture 2: MLP N-gram
**Reimplement and study**: I continue to improve the code from the previous lecture. I first rewind it randomly, and then  it: 1) 80% on a train set to train the neural network 2) 10% validation set to configure hyperparameters 3) 10% test set to check performance at the end on data that the neural network has not seen. I make a multilayer perceptron (MLP) character-level language model. I make the number of characters for a context be chosen in one place. I split the data into training, validation, and test. Learning rate and hyperparameters tuning, under/overfitting, size of network for some number of data. Lookup table of embeddings, minibatch, stochastic gradient descent, size of embeddings, efficient concatenation of embeddings. Learning rate decay. Linear projection S [from an paper by Bengio et al](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) which looks like a residual(skip) connection, but is not. Tracking logs, what degree of error we should expect, how to initialize logits so that the loss is reduced during initialization. A sample of the trained model. The structure of embeddings from the lookup table (it is definitely not random). MY TEST LOSS = 2.1304

**Exercises**:
- [Jupyter file 2](https://github.com/olegdavydovai/reimplement-nn-zero-to-hero/blob/main/lectures/lecture_2_MLP_fourgram.ipynb)

## Lecture 3: Activations & Gradients
**Reimplement and study**: Set initial logits to approximately zero for good initial loss (reduce weights and biases in the last layer to almost 0). In general, we want the activations in the entire neural network to be approximately equal to the standard deviation (Kaiming initialization). Variance, std and histograms of activations. But for deep networks, it is difficult to control activations and normalization layers help with this. Batchnorm layer: running mean and std, gain and bias as parameters of network. In general, the rule of "layer order" works: 1) Linear layer with bias=False (or CNN) 2) Normalization layer 3) Nonlinearity. Tracking and preventing dead neurons that do not receive gradients, vanishing and exploads gradients, simularity break(don't initialize weights = 0).

**Exercises**:
