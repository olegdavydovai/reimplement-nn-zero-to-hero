# Reimplement Neural Networks: Zero to Hero
I watch the playlist on YouTube [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy) - Neural Networks: Zero to hero. Then I recreate the code from the lectures from scratch and do exercises for the lectures. A course on neural networks that starts all the way at the basics. The Jupyter notebooks I build are then captured here inside the [lectures](https://github.com/olegdavydovai/reimplement-nn-zero-to-hero/tree/main/lectures) directory. More details about each completed project are written below ↓↓↓. [names.txt](https://github.com/olegdavydovai/reimplement-nn-zero-to-hero/blob/main/names.txt) is the data for most of the course. [Here](https://github.com/karpathy/nn-zero-to-hero?tab=readme-ov-file) is the original GitHub of the course.

## Table of Contents
- [Lecture 0: Autograd, basics of neural networks, backpropagation](#lecture-0-autograd-basics-of-neural-networks-backpropogation)
- [Lecture 1: Linear Trigram](#lecture-1-linear-trigram)
- [Lecture 2: MLP N-gram](#lecture-2-mlp-n-gram)
- [Lecture 3: Activations & Gradients, Batchnorm1d & Linear](#lecture-3-activations--gradients-batchnorm1d--linear)
- [Lecture 4: Backpropagation](#lecture-4-backpropagation)

## Lecture 0: Autograd, basics of neural networks, backpropogation
**Reimplement and study**: Derivative is core of backpropagation. I am building a Value, Neuron, Layer, MLP classes without libraries, from scratch in Python. Value accepts data, builds an object for them and tracks the operations that go through the entire code. It knows how any value was obtained, as a result of which operations, and stores this information. Then it builds a topographic graph. Backpropagation is performed using chain rule differentiation and the gradient of each parameter with respect to the loss function is obtained. I update them so that the loss function decreases according to the maximum likelihood estimation rules. Optimization is done using vanilla gradient descent. I am implementing a Neuron class that accepts inputs, initializes the weights and biases for the inputs. A Layer class that consists of several independent neurons that accept the same inputs, but different weights and biases for them. An MLP class that consists of several consecutive layers. Don't forget about zero grad. The nonlinearity I use is tanh, loss is MSE.

**Exercises**: Differentiation of an equation through a set of formulas and through a single one. Define log() and the rest that was included in the Value class. Implementation of logits.grad with pytorch.
- [Jupyter file 0](https://github.com/olegdavydovai/reimplement-nn-zero-to-hero/blob/main/lectures/lecture_0_autograd(micrograd).ipynb)

## Lecture 1: Linear Trigram
**MY TEST LOSS**: 2.3730

**Reimplement and study**: I'm implement a bigram character-level language model, i.e. a neural network model take 1 characters as an input to predict the 2nd one. The dataset consists of people's names. I training model, evaluation it and sample new names from scratch. Intoducing pytorch: tensor, torch broadcasting, one_hot. Loss = the negative log likelihood for classification (cross_entropy) + regularization. Linear layer and matrix multiplication, logits + softmax.

**Exercises**: Upgrade Bigram to Trigram (i.e. a neural network model take 2 characters as an input to predict the 3rd one). Split up the dataset randomly into 80% train set, 10% dev set, 10% test set. Train the bigram and trigram models only on the training set. Evaluate them on dev and test splits. Use the dev set to tune the strength of regularization (my best evaluation loss with lr_regularization = 0.0). Instead of using one_hot pull the row from Weights, use F.cross_entropy with pytorch.
- [Jupyter file 1](https://github.com/olegdavydovai/reimplement-nn-zero-to-hero/blob/main/lectures/lecture_1_trigram_bigram.ipynb)

## Lecture 2: MLP N-gram
**MY TEST LOSS**: 2.1304

**Reimplement and study**: I implement a multilayer perceptron (MLP) character-level language model. I first suffle data randomly: 1) 80% on a train set to train the neural network 2) 10% validation set to configure hyperparameters 3) 10% test set to check performance at the end on data that the neural network has not seen. Context size = 3 (for predict 4th). Learning rate and hyperparameters tuning, under/overfitting, size of network for some number of data. Lookup table of embeddings, minibatch, SGD optimizer, size of embeddings, efficient concatenation of embeddings, learning rate decay. Tracking logs, what degree of error we should expect. A evaluate and sample of the trained model. The structure of embeddings from the lookup table (it is definitely not random). Internals pytorch: storage, view, F.cross_entropy.

**Exercises**: Beat validation loss = 2.1701, my validation loss: 2.1289. Tunable hyperparameters: Number of hidden neurons, words in the input, optimization steps, learning rate and decay, dimensionality of embeddings, batch size, convergence time. I changed all hyperparameters and looked at the validation loss, except for the number of optimization steps because the convergence time increased + it would be too easy to achieve a better result. How to initialize logits so that the loss is reduced during initialization (Reduce the weights and offsets of the last layer to almost zero. Expected error: -log(1/vocab_size) = 3.2958). Linear projection S [from an paper by Bengio et al](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) which looks like a residual(skip) connection, but is not.
- [Jupyter file 2](https://github.com/olegdavydovai/reimplement-nn-zero-to-hero/blob/main/lectures/lecture_2_MLP_fourgram.ipynb)

## Lecture 3: Activations & Gradients, Batchnorm1d & Linear
**MY TEST LOSS**: 2.1057

**Reimplement and study**: Statistics of Activations & Gradients throught all neural net. Diagnostic tools and visualizations with matplotlib: activations, grads layer.out and params, ratio: update/data over time. In general, we want the activations in the entire neural network to be approximately equal to the standard deviation (Kaiming initialization). Variance, std and histograms of activations. But for deep networks, it is difficult to control activations and normalization layers help with this. Batchnorm layer: running mean and std as buffers, gamma and beta as parameters of network. In general, the rule of "layer order" works: 1) Linear layer with bias=False (or CNN) 2) Normalization layer 3) Nonlinearity. Tracking and preventing dead neurons that do not receive gradients, vanishing and exploads gradients, simularity break(don't initialize weights = 0). Another way to tune learning rate it's check ratio: update/data = 10**-3. I initializing weights and biases of Linear layer 2 variant: 1) like torch.nn.Linear (uniform distribution) = U(-1/sqrt(fan_in), 1/sqrt(fan_in)) 2) like Andrej's video (normal distribution) = N (0, 1/sqrt(fan_in)). Set initial logits to approximately zero for good initial loss (reduce weights and biases in the last layer to almost 0).

**Exercises**: I collapsed the parameters of Batchnorm1d to the previous Linear when evaluating + if all or all Linear parameters are set to zero, then only the bias (or beta) of the last layer is learned.
- [Jupyter file 3](https://github.com/olegdavydovai/reimplement-nn-zero-to-hero/blob/main/lectures/lecture_3_Activations%20%26%20Gradients%2C%20Batchnorm1d%20%26%20Linear.ipynb)

## Lecture 4: Backpropagation
**MY TEST LOSS**: 2.0899

**Reimplement and study**: Manual backpropagation for tensors to understand it intuitively (via F.cross_entropy, Batchnorm1d, Linear). Analytically derived the formula for F.cross_enropy and Batchnorm1d to calculate the gradients with a single formula. Training a neural network without using torch backward() but with a self-made backward pass. Bessel's correction.

**Exercises**: Output gradients manually without using .backward() and analytically differentiation without the help of lecture.

- [Jupyter file 4](https://github.com/olegdavydovai/reimplement-nn-zero-to-hero/blob/main/lectures/lecture_4_Backpropagation.ipynb)

## Lecture 5: Wavenet without Conv net, Seqential container
**MY TEST LOSS**: -

**Reimplement and study**: Implementation Wavenet without dilated causal convolutions, Seqential, Embedding, Flatten classes. Robust noise loss plot. Containers, nn.Module, submodules.

**Exercises**: Beat 1.993 val loss -

- [Jupyter file 5](https://github.com/olegdavydovai/reimplement-nn-zero-to-hero/blob/main/lectures/lecture_5_Wavenet.ipynb)
