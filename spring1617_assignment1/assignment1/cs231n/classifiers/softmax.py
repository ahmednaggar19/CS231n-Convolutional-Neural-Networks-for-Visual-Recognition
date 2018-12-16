import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize important values
  num_train = X.shape[0] # N
  C = num_class = W.shape[1] # C
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
      scores = X[i].dot(W) # scores using f(W,x) = Wx
      scores -= np.max(scores) # shifting all scores by the maximum value

      loss -= scores[y[i]]

      sum_exp = 0.0
      for score in scores:
        sum_exp += np.exp(score)

      for j in range(C):
        dW[:,j] += ((1.0 / sum_exp) * (np.exp(scores[j]) * X[i]))
        if j == y[i]:
          dW[:,j] -= X[i]
      loss += np.log(sum_exp)

  loss /= num_train
  dW /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = X.dot(W) # all scores (N x C)
  f_exp = np.exp(f)
  loss += (-1 * np.sum(f[range(X.shape[0]), y])) + np.log(np.sum(f_exp))

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  probabilities = np.exp(f) / np.sum(np.exp(f),axis=1,keepdims=True)
  probabilities[range(num_train),y] -= 1
  dW = X.T.dot(probabilities) / num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

# D = 4
# C = 8
# N = 10
# W = np.random.randn(D,C)
# X = np.random.randn(N,D)
# y = np.random.randint(0, high=C, size=N)
# reg = 10
# tuple1 = softmax_loss_naive(W, X, y, reg)
# tuple2 = softmax_loss_vectorized(W, X, y, reg)
# print(tuple1[0] , "\n" ,tuple1[1] ,"\n", tuple2[0], "\n", tuple2[1])
