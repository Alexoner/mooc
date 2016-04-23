#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def activation_statistics(init_func=lambda fan_in, fan_out: np.random.randn(fan_in, fan_out) * 0.001, nonlinearity='tanh'):
    """TODO: Docstring for activation_statistics.
    Demonstrate activation statistics with different weight initialization

    :init_func: TODO
    :returns: TODO

    """
    # assume some uni gaussian 10-D input data
    D = np.random.randn(1000, 500)
    hidden_layer_sizes = [500]*10
    nonlinearities = [nonlinearity]*len(hidden_layer_sizes)

    act = {'relu': lambda x: np.maximum(0,x), 'tanh': lambda x: np.tanh(x)}
    Hs = {}
    for i in range(len(hidden_layer_sizes)):
        X = D if i == 0 else Hs[i-1] # input at this layer
        fan_in = X.shape[1]
        fan_out = hidden_layer_sizes[i]
        W = init_func(fan_in, fan_out) # layer initialization

        H = np.dot(X, W) # matrix multiply
        H = act[nonlinearities[i]](H) # nonlinearities
        Hs[i] = H # cache result on this layer

    # look at the distribution at each layer
    print('input layer had mean %f and std %f' %(np.mean(D), np.std(D)))
    layer_means = [np.mean(H) for i, H in Hs.iteritems()]
    layer_stds = [np.std(H) for i, H in Hs.iteritems()]
    for i, H in Hs.iteritems():
        print('hidden layer %d had mean %f and std %f' % (i+1, layer_means[i], layer_stds[i]))

    # plot the means and standard deviations
    plt.figure()
    plt.subplot(121)
    plt.plot(Hs.keys(), layer_means, 'ob-')
    plt.title('layer mean')
    plt.subplot(122)
    plt.plot(Hs.keys(), layer_stds, 'or-')
    plt.title('layer std')

    # plot the raw distribution
    plt.figure()
    for i,H in Hs.iteritems():
        plt.subplot(2, len(Hs)/2, i+1)
        plt.hist(H.ravel(), 30, range=(-1,1,))
