import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  (C, D) = W.shape
  N  = X.shape[1]
  scores = W.dot(X)
  T = np.zeros((C, N))
  T[y, np.arange(N)] = 1
  for i in xrange(N):
    try:
      scores_i = scores[:, i]
      P_i = softmax(scores_i)
      logP_i = np.log(P_i)
      loss_i = -logP_i[y[i]]
      dloss_i = np.dot((P_i - T[:, i]). reshape((C, 1)), X[:, i].reshape((1, D)))
      loss += loss_i
      dW += dloss_i
    except Exception as e:
      raise e
    else:
      pass

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= N
  dW /= N

  # regularization term
  RW = 0.5 * reg * np.sum(W * W)
  dRW = reg * W
  loss += RW
  dW += dRW
  pass
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  (D, N) = X.shape
  C = W.shape[0]

  # C x N matrix, the linear (inner/dot) product
  Z = W.dot(X)
  # the target value
  P = softmax(Z, axis=0)
  # multiplied by 1-of-K coding target value to get cross-entropy loss
  T = np.zeros_like(P)
  T[y, np.arange(N)] = 1
  loss = np.sum(-np.log(P)[y, np.arange(N)])
  dW = (P - T).dot(X.T)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= N
  dW /= N

  # regularization term
  RW = 0.5 * reg * np.sum(W * W)
  dRW = reg * W
  loss += RW
  dW += dRW
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax(Z, axis=0):
  # the softmax function
  Z_max = np.max(Z, axis=axis)
  Z_normalized = Z - Z_max
  logP = Z_normalized - np.log(np.sum(np.exp(Z_normalized), axis=axis))
  P = np.exp(logP)
  return P
