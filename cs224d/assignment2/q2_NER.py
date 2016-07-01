import os
import getpass
import sys
import time

import numpy as np
import tensorflow as tf
from q2_initialization import xavier_weight_init
import data_utils.utils as du
import data_utils.ner as ner
from utils import data_iterator
from model import LanguageModel

# dropout: 0.500000, lr: 0.0008725245572030131, l2: 1.722356192481772e-05

class Config(object):
  """Holds model hyperparams and data information.

  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
  embed_size = 50
  batch_size = 64
  label_size = 5
  hidden_size = 100
  max_epochs = 2
  early_stopping = 2
  dropout = 0.900000  # well, this is an important hyperparameter. The over-fitting gets heavier as the training epoch moves forward
  lr = 0.003615
  l2 = 0.000035
  window_size = 5

  def __str__(self):
    return '''embed_size: %d, batch_size: %d, label_size: %d, hidden_size: %d,
max_epochs: %d, early_stopping: %d, dropout: %f, lr: %f, l2: %f,
window_size: %d''' % (self.embed_size, self.batch_size, self.label_size,
                              self.hidden_size, self.max_epochs, self.early_stopping,
                              self.dropout, self.lr, self.l2, self.window_size)

class NERModel(LanguageModel):
  """Implements a NER (Named Entity Recognition) model.

  This class implements a deep network for named entity recognition. It
  inherits from LanguageModel, which has an add_embedding method in addition to
  the standard Model method.
  """

  def load_data(self, debug=False):
    """Loads starter word-vectors and train/dev/test data."""
    # Load the starter word vectors
    self.wv, word_to_num, num_to_word = ner.load_wv(
      'data/ner/vocab.txt', 'data/ner/wordVectors.txt')
    tagnames = ['O', 'LOC', 'MISC', 'ORG', 'PER']
    self.num_to_tag = dict(enumerate(tagnames))
    tag_to_num = {v:k for k,v in self.num_to_tag.items()}

    # Load the training set
    docs = du.load_dataset('data/ner/train')
    self.X_train, self.y_train = du.docs_to_windows(
        docs, word_to_num, tag_to_num, wsize=self.config.window_size)
    if debug:
      self.X_train = self.X_train[:1024]
      self.y_train = self.y_train[:1024]

    # Load the dev set (for tuning hyperparameters)
    docs = du.load_dataset('data/ner/dev')
    self.X_dev, self.y_dev = du.docs_to_windows(
        docs, word_to_num, tag_to_num, wsize=self.config.window_size)
    if debug:
      self.X_dev = self.X_dev[:1024]
      self.y_dev = self.y_dev[:1024]

    # Load the test set (dummy labels only)
    docs = du.load_dataset('data/ner/test.masked')
    self.X_test, self.y_test = du.docs_to_windows(
        docs, word_to_num, tag_to_num, wsize=self.config.window_size)

  def add_placeholders(self):
    """Generate placeholder variables to represent the input tensors

    These placeholders are used as inputs by the rest of the model building
    code and will be fed data during training.  Note that when "None" is in a
    placeholder's shape, it's flexible

    Adds following nodes to the computational graph

    input_placeholder: Input placeholder tensor of shape
                       (None, window_size), type tf.int32
    labels_placeholder: Labels placeholder tensor of shape
                        (None, label_size), type tf.float32
    dropout_placeholder: Dropout value placeholder (scalar),
                         type tf.float32

    Add these placeholders to self as the instance variables

      self.input_placeholder
      self.labels_placeholder
      self.dropout_placeholder

    (Don't change the variable names)
    """
    ### TODO: YOUR CODE HERE
    self.input_placeholder = tf.placeholder(
      tf.int32, shape=(None, self.config.window_size), name='Input')
    self.labels_placeholder = tf.placeholder(
      tf.float32, shape=(None, self.config.label_size), name='Labels')
    self.dropout_placeholder = tf.placeholder(tf.float32, name='Dropout')
    ### END YOUR CODE

  def create_feed_dict(self, input_batch, dropout, label_batch=None):
    """Creates the feed_dict for softmax classifier.

    A feed_dict takes the form of:

    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }


    Hint: The keys for the feed_dict should be a subset of the placeholder
          tensors created in add_placeholders.
    Hint: When label_batch is None, don't add a labels entry to the feed_dict.

    Args:
      input_batch: A batch of input data.
      label_batch: A batch of label data.
    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    ### TODO: YOUR CODE HERE
    feed_dict = {
        self.input_placeholder: input_batch,
    }
    if label_batch is not None:
        feed_dict[self.labels_placeholder] = label_batch
    if dropout is not None:
        feed_dict[self.dropout_placeholder] = dropout
    ### END YOUR CODE
    return feed_dict

  def add_embedding(self):
    """Add embedding layer that maps from vocabulary to vectors.

    Creates an embedding tensor (of shape (len(self.wv), embed_size). Use the
    input_placeholder to retrieve the embeddings for words in the current batch.

    (Words are discrete entities. They need to be transformed into vectors for use
    in deep-learning. Although we won't do so in this problem, in practice it's
    useful to initialize the embedding with pre-trained word-vectors. For this
    problem, using the default initializer is sufficient.)

    Hint: This layer should use the input_placeholder to index into the
          embedding.
    Hint: You might find tf.nn.embedding_lookup useful.
    Hint: See following link to understand what -1 in a shape means.
      https://www.tensorflow.org/versions/r0.8/api_docs/python/array_ops.html#reshape
    Hint: Check the last slide from the TensorFlow lecture.
    Hint: Here are the dimensions of the variables you will need to create:

      L: (len(self.wv), embed_size)

    Returns:
      window: tf.Tensor of shape (-1, window_size*embed_size)
    """
    # The embedding lookup is currently only implemented for the CPU
    with tf.device('/cpu:0'):
      ### TODO: YOUR CODE HERE
      embeddings = tf.Variable(
          tf.random_uniform((len(self.wv), self.config.embed_size), -1.0, 1.0), name='Embeddings')
      window = tf.nn.embedding_lookup(embeddings, self.input_placeholder)
      window = tf.reshape(window, (-1, self.config.window_size * self.config.embed_size))
      ### END YOUR CODE
      return window

  def add_model(self, window):
    """Adds the 1-hidden-layer NN.

    Hint: Use a variable_scope (e.g. "Layer") for the first hidden layer, and
          another variable_scope (e.g. "Softmax") for the linear transformation
          preceding the softmax. Make sure to use the xavier_weight_init you
          defined in the previous part to initialize weights.
    Hint: Make sure to add in regularization and dropout to this network.
          Regularization should be an addition to the cost function, while
          dropout should be added after both variable scopes.
    Hint: You might consider using a tensorflow Graph Collection (e.g
          "total_loss") to collect the regularization and loss terms (which you
          will add in add_loss_op below).
    Hint: Here are the dimensions of the various variables you will need to
          create

          W:  (window_size*embed_size, hidden_size)
          b1: (hidden_size,)
          U:  (hidden_size, label_size)
          b2: (label_size)

    https://www.tensorflow.org/versions/r0.7/api_docs/python/framework.html#graph-collections
    Args:
      window: tf.Tensor of shape (-1, window_size*embed_size)
    Returns:
      output: tf.Tensor of shape (batch_size, label_size)
    """
    ### TODO: YOUR CODE HERE
    xavier_initializer = xavier_weight_init()
    with tf.variable_scope('Layer', initializer=xavier_initializer) as scope:
        W = tf.get_variable('W',
                            shape=(self.config.window_size * self.config.embed_size,
                                   self.config.hidden_size)
                            )
        b1 = tf.Variable(tf.zeros((self.config.hidden_size,)))
        if self.config.l2:
          tf.add_to_collection('total_loss', self.config.l2 * tf.nn.l2_loss(W))
    a1 = tf.nn.tanh(tf.matmul(window, W) + b1)
    with tf.variable_scope('Softmax', initializer=xavier_initializer) as scope:
        U = tf.get_variable('U',
                            shape=(self.config.hidden_size, self.config.label_size),
                            )
        b2 = tf.zeros((self.config.label_size))
        if self.config.l2:
          tf.add_to_collection('total_loss', 0.5 * self.config.l2 * tf.reduce_sum(U ** 2))
    output = tf.nn.dropout(tf.matmul(a1, U) + b2, self.dropout_placeholder)
    ### END YOUR CODE
    return output

  def add_loss_op(self, y):
    """Adds cross_entropy_loss ops to the computational graph.

    Hint: You can use tf.nn.softmax_cross_entropy_with_logits to simplify your
          implementation. You might find tf.reduce_mean useful.
    Args:
      pred: A tensor of shape (batch_size, n_classes)
    Returns:
      loss: A 0-d tensor (scalar)
    """
    ### TODO: YOUR CODE HERE
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        y, self.labels_placeholder))
    loss += tf.reduce_sum(tf.get_collection('total_loss'))
    ### END YOUR CODE
    return loss

  def add_training_op(self, loss):
    """Sets up the training Ops.

    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train. See

    https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

    for more information.

    Hint: Use tf.train.AdamOptimizer for this model.
          Calling optimizer.minimize() will return a train_op object.

    Args:
      loss: Loss tensor, from cross_entropy_loss.
    Returns:
      train_op: The Op for training.
    """
    ### TODO: YOUR CODE HERE
    opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)
    train_op = opt.minimize(loss)
    ### END YOUR CODE
    return train_op

  def __init__(self, config):
    """Constructs the network using the helper functions defined above."""
    self.config = config
    self.load_data(debug=False)
    self.add_placeholders()
    window = self.add_embedding()
    y = self.add_model(window)

    self.loss = self.add_loss_op(y)
    self.predictions = tf.nn.softmax(y)
    one_hot_prediction = tf.argmax(self.predictions, 1)
    correct_prediction = tf.equal(
        tf.argmax(self.labels_placeholder, 1), one_hot_prediction)
    self.correct_predictions = tf.reduce_sum(tf.cast(correct_prediction, 'int32'))
    self.train_op = self.add_training_op(self.loss)

  def run_epoch(self, session, input_data, input_labels,
                shuffle=True, verbose=True):
    orig_X, orig_y = input_data, input_labels
    dp = self.config.dropout
    # We're interested in keeping track of the loss and accuracy during training
    total_loss = []
    total_correct_examples = 0
    total_processed_examples = 0
    total_steps = len(orig_X) / self.config.batch_size
    for step, (x, y) in enumerate(
      data_iterator(orig_X, orig_y, batch_size=self.config.batch_size,
                   label_size=self.config.label_size, shuffle=shuffle)):
      feed = self.create_feed_dict(input_batch=x, dropout=dp, label_batch=y)
      loss, total_correct, _ = session.run(
          [self.loss, self.correct_predictions, self.train_op],
          feed_dict=feed)
      total_processed_examples += len(x)
      total_correct_examples += total_correct
      total_loss.append(loss)
      ##
      if verbose and step % verbose == 0:
        sys.stdout.write('\r{} / {} : loss = {}'.format(
          step, total_steps, np.mean(total_loss)))
        sys.stdout.flush()
    if verbose:
        sys.stdout.write('\r')
        sys.stdout.flush()
    return np.mean(total_loss), total_correct_examples / float(total_processed_examples)

  def predict(self, session, X, y=None):
    """Make predictions from the provided model."""
    # If y is given, the loss is also calculated
    # We deactivate dropout by setting it to 1
    dp = 1
    losses = []
    results = []
    if np.any(y):
        data = data_iterator(X, y, batch_size=self.config.batch_size,
                             label_size=self.config.label_size, shuffle=False)
    else:
        data = data_iterator(X, batch_size=self.config.batch_size,
                             label_size=self.config.label_size, shuffle=False)
    for step, (x, y) in enumerate(data):
      feed = self.create_feed_dict(input_batch=x, dropout=dp)
      if np.any(y):
        feed[self.labels_placeholder] = y
        loss, preds = session.run(
            [self.loss, self.predictions], feed_dict=feed)
        losses.append(loss)
      else:
        preds = session.run(self.predictions, feed_dict=feed)
      predicted_indices = preds.argmax(axis=1)
      results.extend(predicted_indices)
    return np.mean(losses), results

def print_confusion(confusion, num_to_tag):
    """Helper method that prints confusion matrix."""
    # Summing top to bottom gets the total number of tags guessed as T
    total_guessed_tags = confusion.sum(axis=0)
    # Summing left to right gets the total number of true tags
    total_true_tags = confusion.sum(axis=1)
    print()
    print(confusion)
    for i, tag in sorted(num_to_tag.items()):
        prec = confusion[i, i] / float(total_guessed_tags[i])
        recall = confusion[i, i] / float(total_true_tags[i])
        print('Tag: {} - P {:2.4f} / R {:2.4f}'.format(tag, prec, recall))

def calculate_confusion(config, predicted_indices, y_indices):
    """Helper method that calculates confusion matrix."""
    confusion = np.zeros((config.label_size, config.label_size), dtype=np.int32)
    for i in range(len(y_indices)):
        correct_label = y_indices[i]
        guessed_label = predicted_indices[i]
        confusion[correct_label, guessed_label] += 1
    return confusion

def save_predictions(predictions, filename):
  """Saves predictions to provided file."""
  with open(filename, "wb") as f:
    for prediction in predictions:
      f.write((str(prediction) + "\n").encode())

def test_NER():
  """Test NER model implementation.

  You can use this function to test your implementation of the Named Entity
  Recognition network. When debugging, set max_epochs in the Config object to 1
  so you can rapidly iterate.
  """
  config = Config()
  with tf.Graph().as_default():
    model = NERModel(config)

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    with tf.Session() as session:
      best_val_loss = float('inf')
      best_val_epoch = 0

      session.run(init)
      for epoch in range(config.max_epochs):
        print('Epoch {}'.format(epoch))
        start = time.time()
        ###
        train_loss, train_acc = model.run_epoch(session, model.X_train,
                                                model.y_train)
        val_loss, predictions = model.predict(session, model.X_dev, model.y_dev)
        print('Training loss: {}'.format(train_loss))
        print('Training acc: {}'.format(train_acc))
        print('Validation loss: {}'.format(val_loss))
        if val_loss < best_val_loss:
          best_val_loss = val_loss
          best_val_epoch = epoch
          if not os.path.exists("./weights"):
            os.makedirs("./weights")

          saver.save(session, './weights/ner.weights')
        if epoch - best_val_epoch > config.early_stopping:
          break
        ###
        confusion = calculate_confusion(config, predictions, model.y_dev)
        print_confusion(confusion, model.num_to_tag)
        print('Total time: {}'.format(time.time() - start))

      saver.restore(session, './weights/ner.weights')
      print('Test')
      print('=-=-=')
      print('Writing predictions to q2_test.predicted')
      _, predictions = model.predict(session, model.X_test, model.y_test)
      save_predictions(predictions, "q2_test.predicted")

def run_model(config):
  with tf.Graph().as_default():
    model = NERModel(config)

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    best_val_loss = float('inf')
    best_val_epoch = 0

    with tf.Session() as session:
        session.run(init)
        for epoch in range(config.max_epochs):
            print('Epoch {}'.format(epoch))
            start = time.time()
            ###
            train_loss, train_acc = model.run_epoch(session, model.X_train,
                                                    model.y_train)
            val_loss, predictions = model.predict(session, model.X_dev, model.y_dev)
            print('Training loss: {}'.format(train_loss))
            print('Training acc: {}'.format(train_acc))
            print('Validation loss: {}'.format(val_loss))
            if val_loss < best_val_loss:
              best_val_loss = val_loss
              best_val_epoch = epoch
              if not os.path.exists("./weights"):
                os.makedirs("./weights")

              saver.save(session, './weights/ner.weights')
            if epoch - best_val_epoch > config.early_stopping:
              break
            ###
            confusion = calculate_confusion(config, predictions, model.y_dev)
            print_confusion(confusion, model.num_to_tag)
            print('Total time: {}'.format(time.time() - start))

    return train_loss, train_acc, best_val_loss

def cross_validate():
  """ Cross validate to get best performance
  """
  embed_sizes = [50, 100][:1]
  batch_sizes = [64, 128][:1]
  hidden_sizes = [50, 100, 150][1:2]
  window_sizes = [3, 4, 5][2:]
  dropouts = [0.5, 0.7, 0.9]
  lrs = sorted(10 ** np.random.uniform(-2.5, -2, 2))
  l2s = sorted(10 ** np.random.uniform(-4.4, -3, 2))

  config = Config()
  config.max_epochs = 2
  results = []
  from itertools import product
  for (embed_size, batch_size, hidden_size, window_size, dropout, lr, l2
  ) in product(embed_sizes, batch_sizes, hidden_sizes, window_sizes, dropouts, lrs, l2s):
    config.embed_size = embed_size
    config.batch_size = batch_size
    config.hidden_size = hidden_size
    config.window_size = window_size
    config.dropout = dropout
    config.lr = lr
    config.l2 = l2

    print('config: ', config)
    results.append((embed_size, batch_size, hidden_size, window_size, lr, l2) + \
                   run_model(config))
    print('===============================================================\n')

  results = sorted(results, key=lambda x: x[-1])
  print(results)
  best_config = results[-1]

if __name__ == "__main__":
  if len(sys.argv) > 1:
    sys.stdout.write('running cross validation\n')
    sys.stdout.flush()
    cross_validate()
  else:
    test_NER()
