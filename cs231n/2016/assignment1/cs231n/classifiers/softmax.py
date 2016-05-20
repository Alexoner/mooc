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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  (D, C) = W.shape
  N  = X.shape[0]
  # N x C matrix
  scores = X.dot(W)
  for i in xrange(N):
    scores_i = scores[i, :]
    scores_i -= np.max(scores_i, axis=-1)
    P_i = np.exp(scores_i)
    P_i /= np.sum(P_i, axis=-1)
    logP_i = np.log(P_i)
    loss_i = -logP_i[y[i]]
    dscores_i = P_i
    dscores_i[y[i]] -= 1
    #  dscores_i = np.dot((P_i - T[:, i]). reshape((C, 1)), X[:, i].reshape((1, D)))
    loss += loss_i
    dW +=  X[i, :].reshape((D, 1)).dot(dscores_i.reshape((1, C)))

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
  num_train, D = X.shape
  scores = X.dot(W) # N x C

  scores -= np.max(scores, axis=1, keepdims=True)
  p = np.exp(scores)
  p /= np.sum(p, axis=1, keepdims=True)

  # cross-entropy loss
  loss_data = -np.sum(np.log(p[range(y.size), y])) / num_train
  loss_reg = 0.5 * reg * np.sum(W * W)
  loss = loss_data + loss_reg

  dscores = p
  dscores[range(y.size), y] -= 1.0
  dW = X.T.dot(dscores) / num_train + reg * W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

