import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0] # number of classes
  num_train = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes): # iterate all wrong classes
      if j == y[i]:
        # skip the true class to only loop over incorrect classes
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        # accumulate loss for the i-th example
        loss += margin
        dW[y[i], :] += -X[:, i].transpose()
        dW[j] += X[:, i].transpose()

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += 0.5 * reg * (2 * W)

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
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  delta = 1.0
  num_classes = W.shape[0]
  num_train = X.shape[1]
  # C x N matrix
  scores = W.dot(X)
  # 1 x N vector, correct class scores
  correct_class_score = scores[y, np.arange(0, num_train)]

  # compute the margins for all classes in on vector operation
  # C x N matrix
  margins = np.maximum(0, scores - correct_class_score + delta)
  # on y-th position scores[y] - scores[y] canceled and gave delta. We want
  # to ignore the y-th position and only consider margin on max wrong class
  margins[y, np.arange(0, num_train)] = np.zeros(( 1, num_train ))

  loss = np.sum(margins)
  pass
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
  dmargins                          = np.zeros(margins.shape)
  dmargins[y, np.arange(num_train)] = -np.sum(margins>0, axis=0)
  dmargins[margins>0]              = 1
  dW = dmargins.dot(X.T)
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  # regularized term
  RW = 0.5 * reg * np.sum(W * W)
  dRW = 0.5 * reg * (2 * W)
  loss += RW
  dW += dRW

  return loss, dW
