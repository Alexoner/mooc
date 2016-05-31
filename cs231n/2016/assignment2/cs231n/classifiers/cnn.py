import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    self.params['W1'] = weight_scale * np.random.randn(
        num_filters, C, filter_size, filter_size)
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = weight_scale * np.random.randn(
        num_filters * H * W / 4, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = weight_scale * np.random.randn(
        hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    a1, cache1 = conv_relu_pool_forward(
        X, self.params['W1'], self.params['b1'], conv_param, pool_param)
    a2, cache2 = affine_relu_forward(a1, self.params['W2'], self.params['b2'])
    a3, cache3 = affine_forward(a2, self.params['W3'], self.params['b3'])
    scores = a3
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss_data, dscores            = softmax_loss(scores, y)
    da3                           = dscores
    da2, grads['W3'], grads['b3'] = affine_backward(da3, cache3)
    da1, grads['W2'], grads['b2'] = affine_relu_backward(da2, cache2)
    dX, grads['W1'], grads['b1']  = conv_relu_pool_backward(da1, cache1)

    loss_reg = 0.0
    for p in ['W1', 'W2', 'W3']:
      loss_reg += 0.5 * self.reg * np.sum(self.params[p] ** 2)
      grads[p] += self.reg * self.params[p]
    loss = loss_data + loss_reg
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FunConvNet(object):
  """
  A customized convolutional network for fun, with the following architecture:

  { conv - [spatial batch norm] - relu - 2x2 max pool } x N -
    { [dropout] - affine - [relu] } x M - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.  """

  def __init__(self, input_dim=(3, 32, 32),
               conv_params=[
                   {
                       'num_filters': 64, 'filter_size': 5,
                       'stride': 1, #'pad': (filter_size - 1) / 2
                   },
                   {
                       'num_filters': 128, 'filter_size': 3,
                       'stride': 1, #'pad': (filter_size - 1) / 2
                   },
               ],
               hidden_dims=[512, 128], num_classes=10,
               dropout=0, use_batchnorm=False,
               weight_scale=1e-3, reg=0.0,
               dtype=np.float32, seed=None):
    """
    Initialize a new network.

    Inputs:
    - input_dim    : Tuple (C, H, W) giving size of input data
    - conv_params  : convolutional layers parameters
        - num_filters  : Number of filters to use in the convolutional layer
        - filter_size : Sequence of size of filters to use in the convolutional layer
        - stride : window moving stride
        - pad: default (filter_size -1)/2 to preserve the previous dimension
    - pool_params  : not exposed yet!
    - hidden_dims  : A list of integers number of units to use in the
      fully-connected hidden layer.
    - num_classes : Number of scores to produce from the final affine layer.
    - dropout     : Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.Drop the neurons at a probability
      of dropout argument.
    - use_batchnorm : Whether or not the network should use batch normalization.
    - weight_scale  : Scalar giving standard deviation for random initialization
      of weights.
    - reg   : Scalar giving L2 regularization strength
    - dtype : numpy datatype to use for computation.
    - seed  : If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriministic so we can gradient check the
      model.
    """
    self.params = {}
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg

    self.conv_params = conv_params
    self.num_convs = len(conv_params)
    self.hidden_dims = hidden_dims
    # plus one last affine layer producing output scores
    self.num_layers = self.num_convs + len(hidden_dims) + 1
    self.dtype = dtype

    self.bn_params = use_batchnorm and [
        {'mode': 'train'} for l in range(self.num_convs)] or None
    self.pool_params = [{'pool_height': 2, 'pool_width': 2, 'stride': 2}
                        for l in range(self.num_convs)]
    self.dropout_params = dropout and \
            [{'mode': 'train', 'p': dropout}
             for l in range(len(self.hidden_dims))] or None

    ############################################################################
    # TODO: Initialize weights and biases for the fun convolutional            #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    C, H, W = input_dim
    for l in range(self.num_layers):
      l1 = l + 1
      # conv layers
      if l < self.num_convs:
        # meta parameters
        self.conv_params[l].setdefault(
            'pad', (self.conv_params[l]['filter_size']-1 )/2)

        # learnable parameters
        # TODO: consider use weight np.sqrt(2.0 / fan_in) here
        self.params['W%d' % l1] = weight_scale * np.random.randn(
          self.conv_params[l]['num_filters'],
          l and self.conv_params[l-1]['num_filters'] or C,
          self.conv_params[l]['filter_size'], self.conv_params[l]['filter_size']
        )
        self.params['b%d' % l1] = np.zeros((self.conv_params[l]['num_filters']))
        if self.bn_params:
          self.params['gamma%d' % l1] = np.ones((self.conv_params[l]['num_filters']))
          self.params['beta%d' % l1] = np.zeros((self.conv_params[l]['num_filters']))
        pass
      # fully connected layers
      elif l < self.num_layers - 1:
          if seed:
            self.dropout_params[l-self.num_convs]['seed'] = seed

          self.params['W%d' % l1] = weight_scale * np.random.randn(
            l == self.num_convs and
              self.conv_params[l-1]['num_filters'] * H * W * 2 ** (-2 * self.num_convs)
              or self.hidden_dims[l - 1 - self.num_convs],
            self.hidden_dims[l - self.num_convs])
          self.params['b%d' % l1] = np.zeros(self.hidden_dims[l - self.num_convs])
      # last affine layer
      else:
          self.params['W%d' % l1] = weight_scale * np.random.randn(
            l == self.num_convs and self.conv_params[l-1]['num_filters'] * H * W
              or self.hidden_dims[l - 1 - self.num_convs],
            num_classes)
          self.params['b%d' % l1] = np.zeros(num_classes)
          pass

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'


    # pass conv_param to the forward pass for the convolutional layer

    scores = None
    ############################################################################
    # TODO: Implement the FORWARD pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################

    # forward pass
    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_params:
      for dropout_param in self.dropout_params:
        dropout_param['mode'] = mode
    if self.use_batchnorm and self.bn_params:
      for bn_param in self.bn_params:
        bn_param['mode'] = mode

    # begin to forward
    layer_caches = []
    prev_a = X
    for l in range(self.num_layers):
      l1 = l + 1
      cache = {}
      if l < self.num_convs:
        conv_param = self.conv_params[l]
        bn_param = self.bn_params and self.bn_params[l]
        pool_param = self.pool_params[l]

        prev_a, cache['conv'] = conv_bn_relu_pool_forward(
            prev_a, self.params['W%d' % (l+1)], self.params['b%d' % l1],
            bn_param and self.params['gamma%d' % l1],
            bn_param and self.params['beta%d' % l1],
            conv_param, bn_param, pool_param
        )
      elif l == self.num_layers -1:
        prev_a, cache['affine'] = affine_forward(
            prev_a, self.params['W%d' % l1], self.params['b%d' % l1])
      else:
        # NOTE: if use dropout layers
        if self.dropout_params:
          prev_a, cache['dropout'] = dropout_forward(
              prev_a, self.dropout_params[l - self.num_convs])
        # TODO: if use relu
        # prev_a, cache_affine = affine_relu_forward(
            # prev_a, self.params['W%d' % l1], self.params['b%d' % l1])
        prev_a, cache['affine'] = affine_forward(
            prev_a, self.params['W%d' % l1], self.params['b%d' % l1])
      layer_caches.append(cache)
      pass
    scores = prev_a
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the BACKWARD pass for the convolutional net,             #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################

    # FIXME: this loss implementation still fails gradient checking
    loss_reg = 0.0
    loss_data, dscores = softmax_loss(scores, y)
    dout = dscores
    for l in reversed(range(self.num_layers)):
      l1 = l + 1
      if l < self.num_convs:
        (dout,
         grads['W%d' % l1], grads['b%d' % l1],
         grads['gamma%d' % l1], grads['beta%d' % l1]
         ) = conv_bn_relu_pool_backward(dout, layer_caches[l]['conv'])
        if not self.bn_params:
          del grads['gamma%d' % l1], grads['beta%d' % l1]
      elif l == self.num_layers - 1:
        (dout,
         grads['W%d' % l1],
         grads['b%d' % l1]) = affine_backward(dout, layer_caches[l]['affine'])
      else:
        # TODO: if use relu
        # (dout,
         # grads['W%d' % l1],
         # grads['b%d' % l1]) = affine_relu_backward(
             # dout, layer_caches[l]['affine'])
        (dout,
         grads['W%d' % l1],
         grads['b%d' % l1]) = affine_backward(dout, layer_caches[l]['affine'])
        # NOTE: if use dropout layers
        if self.dropout_params:
          dout = dropout_backward(
              dout, layer_caches[l]['dropout'])

      loss_reg += 0.5 * self.reg * np.sum(self.params['W%d' % l1] ** 2)
      grads['W%d' % l1] += self.reg * self.params['W%d' % l1]

    loss = loss_data + loss_reg
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

  def predict(self, X):
      scores = self.loss(X)
      y_predict = np.argmax(scores, axis=-1)
      return y_predict

pass
