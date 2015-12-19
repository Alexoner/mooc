from main import replace_accented
from sklearn import svm
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
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
                          list(nltk.word_tokenize(t[3]))[0:window_size]
                         ),
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
    vec = DictVectorizer()
    s_set = set(s)
    def vectorize_one(t):
        tokens_left = list(nltk.word_tokenize(t[1]))
        tokens_right = list(nltk.word_tokenize(t[3]))
        tokens = tokens_left + [t[2]] + tokens_right
        context_words = tokens_left[-window_size:] + tokens_right[0:window_size]
        context_window = dict(map(lambda x: ('BOW_'+x, 0), s))
        def inc_one(word):
            if word in s_set:
                key = 'BOW_'+word
                context_window.setdefault(key, 0)
                context_window[key] += 1

        try:
            map(lambda word: inc_one(word),
                context_words
                )
        except Exception as e:
            # print 'word', 'not in s ', e
            pass
        try:
            vectors[t[0]] = context_window
        except:
            pass
        labels[t[0]] = t[-1]
    map(vectorize_one, data)
    vec.fit(vectors.values())
    for instance_id in vectors:
        vectors[instance_id] = vec.transform(vectors[instance_id]).toarray()[0]
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
    #C=100,10,1.0,0.1,0.1,0.01,0.001,0.0001
    # 1.0:0.536, 0.51, 0.1: , 0.01:0.543, 0.005:0.536, 0.075:0.538, 0.015:0.541, 0.011:0.541, 0.0109: 0.540
    svm_clf = svm.LinearSVC(C=1.01, verbose=0, random_state=0)
    # k=4:0.48, 5:0.492, 6: 0.499, 7:0.510, 8:0.518, 9:0.519, 10:0.529, 11:0.532, 14:0.535
    knn_clf = neighbors.KNeighborsClassifier(14)
    # the label encoder seems not necessary
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(y_train.values())

    # implement your code here
    X_train_arr = sparse.csr_matrix(X_train.values())
    # y_train_arr = sparse.csr_matrix(np.array(y_train.values()))
    X_test_arr = sparse.csr_matrix(X_test.values())
    # X_train_arr = np.array(X_train.values())
    y_train_arr = np.array(label_encoder.transform(y_train.values()))
    # X_test_arr = np.array(X_test.values())
    svm_clf.fit(X_train_arr, y_train_arr)
    knn_clf.fit(X_train_arr, y_train_arr)
    svm_predicted = svm_clf.predict(X_test_arr)
    knn_predicted = knn_clf.predict(X_test_arr)
    svm_score = svm_clf.score(X_train_arr, y_train_arr)
    svm_results = zip(X_test.keys(),
                      label_encoder.inverse_transform(svm_predicted))
    knn_results = zip(X_test.keys(),
                      label_encoder.inverse_transform(knn_predicted))

    print "svm score:", svm_score
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
                                 f.write('{0} {1} {2}\n'.format(
                                     replace_accented(item[0]),
                                     replace_accented(t[0]),
                                     replace_accented(unicode(t[1])))),
                                 item[1]),
                map(lambda result: (result[0], sorted(result[1], cmp)),
                    sorted(results.items())))
        except Exception as e:
            print e
            pass

# run part A
def run(train, test, language, knn_file, svm_file):
    print 'running A'
    s = build_s(train)
    svm_results = {}
    knn_results = {}
    for lexelt in s:
        X_train, y_train = vectorize(train[lexelt], s[lexelt])
        X_test, _ = vectorize(test[lexelt], s[lexelt])
        # if lexelt == 'activate.v':
            # pass
        # else:
            # continue
        svm_results[lexelt], knn_results[lexelt] = classify(X_train, X_test, y_train)

    print_results(svm_results, svm_file)
    print_results(knn_results, knn_file)



