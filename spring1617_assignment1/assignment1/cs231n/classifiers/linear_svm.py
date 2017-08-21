import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        dW[:,j] += X[i]
        dW[:,y[i]] -= X[i] 
        loss += margin
        

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  num_train = X.shape[0]
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  nList = np.arange(num_train)
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  all_scores = X.dot(W)
  all_margins = (all_scores.T - all_scores[nList, y] + 1).T # assuming same delta = 1
  all_margins[nList, y] = 0   # do not count correct class scores' margins
  all_margins[all_margins < 0] = 0 # do not count negative margins
  loss += np.sum(all_margins) 
  loss /= num_train
  loss += reg * np.sum( W * W )
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  num_pos = np.sum(all_margins.T > 0, axis=0) # number of positive margins

  dscores = np.zeros(all_scores.T.shape)
  dscores[all_margins.T > 0] = 1
  dscores[y, range(num_train)] = -num_pos

  dW = dscores.dot(X)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW.T

########## Testing ##########
D = 4
C = 8
N = 10
W = np.random.randn(D,C)
X = np.random.randn(N,D)
y = np.random.randint(0, high=C, size=N)
reg = 10
tuple1 = svm_loss_naive(W, X, y, reg)
tuple2 = svm_loss_vectorized(W, X, y, reg)
print(tuple1[1], "\n"
      , tuple2[1])
