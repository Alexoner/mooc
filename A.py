from main import replace_accented
from sklearn import svm
from sklearn import neighbors
from sklearn import preprocessing
import nltk
import numpy as np
import sys
from scipy import sparse
import operator

# don't change the window size
window_size = 10

# A.1
def build_s(data):
    '''
    Compute the context vector for each lexelt
    :param data: dic with the following structure:
        {
			lexelt: [(instance_id, left_context, head, right_context, sense_id), ...],
			...
        }
    :return: dic s with the following structure:
        {
			lexelt: [w1,w2,w3, ...],
			...
        }

    '''
    s = {}

    # implement your code here
    def get_context_words(lexelt):
        words = set()
        map(lambda t: map(lambda w: words.add(w),
                          list(nltk.word_tokenize(t[1]))[-window_size:] +
                          list(nltk.word_tokenize(t[3]))[0:window_size+1]),
            data[lexelt])
        s[lexelt] = list(words)
    map(get_context_words, data)

    return s


# A.1
def vectorize(data, s):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :param s: list of words (features) for a given lexelt: [w1,w2,w3, ...]
    :return: vectors: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }

    '''
    vectors = {}
    labels = {}

    # implement your code here
    def vectorize_one(t):
        context_vector = map(lambda x: 0, s)
        try:
            map(lambda word: operator.setitem(context_vector, s.index(word), context_vector[s.index(word)]+1),
                list(nltk.word_tokenize(t[1]))[-window_size:] +
                list(nltk.word_tokenize(t[3]))[0:window_size+1])
        except Exception as e:
            # print data, s
            # print 'word', 'not in s ', e
            pass
        vectors[t[0]] = context_vector
        labels[t[0]] = t[4]
    map(vectorize_one, data)
    return vectors, labels


# A.2
def classify(X_train, X_test, y_train):
    '''
    Train two classifiers on (X_train, and y_train) then predict X_test labels

    :param X_train: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param X_test: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }

    :return: svm_results: a list of tuples (instance_id, label) where labels are predicted by LinearSVC
             knn_results: a list of tuples (instance_id, label) where labels are predicted by KNeighborsClassifier
    '''

    svm_results = []
    knn_results = []

    # svm_clf = svm.SVC(gamma=0.001, C=1.)
    svm_clf = svm.LinearSVC(C=10.)
    knn_clf = neighbors.KNeighborsClassifier(10, weights='uniform')
    # the label encoder seems not necessary
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(y_train.values())

    # implement your code here
    # X_train_arr = sparse.csr_matrix(np.array(X_train.values()))
    # y_train_arr = sparse.csr_matrix(np.array(y_train.values()))
    # X_test_arr = sparse.csr_matrix(np.array(X_test.values()))
    X_train_arr = np.array(X_train.values())
    y_train_arr = np.array(label_encoder.transform(y_train.values()))
    X_test_arr = np.array(X_test.values())
    svm_clf.fit(X_train_arr, y_train_arr)
    knn_clf.fit(X_train_arr, y_train_arr)
    svm_predicted = svm_clf.predict(X_test_arr)
    knn_predicted = knn_clf.predict(X_test_arr)
    svm_results = zip(X_test.keys(),
                      label_encoder.inverse_transform(svm_predicted))
    knn_results += zip(X_test.keys(),
                       label_encoder.inverse_transform(knn_predicted))

    return svm_results, knn_results

# A.3, A.4 output
def print_results(results, output_file):
    '''

    :param results: A dictionary with key = lexelt and value = a list of tuples (instance_id, label)
    :param output_file: file to write output

    '''

    # implement your code here
    # don't forget to remove the accent of characters using main.replace_accented(input_str)
    # you should sort results on instance_id before printing
    with open(output_file, 'w') as f:
        try:
            map(lambda item: map(lambda t:
                                 f.write(replace_accented(
                                     item[0]+' '+t[0]+' '+t[1]+'\n')),
                                 item[1]),
                map(lambda result: (result[0], sorted(result[1], cmp)),
                    sorted(results.items())))
        except Exception as e:
            print e
            pass

# run part A
def run(train, test, language, knn_file, svm_file):
    s = build_s(train)
    svm_results = {}
    knn_results = {}
    for lexelt in s:
        X_train, y_train = vectorize(train[lexelt], s[lexelt])
        X_test, _ = vectorize(test[lexelt], s[lexelt])
        svm_results[lexelt], knn_results[lexelt] = classify(X_train, X_test, y_train)
        # print 'predicted results:', svm_results[lexelt], knn_results[lexelt]

    print_results(svm_results, svm_file)
    print_results(knn_results, knn_file)



