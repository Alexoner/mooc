import A
from sklearn.feature_extraction import (DictVectorizer,)
from sklearn.feature_selection import (VarianceThreshold,
                                       SelectKBest,
                                       chi2,
                                       )
from sklearn import (svm, neighbors,
                     preprocessing, cross_validation,
                     )
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import KFold
from scipy import sparse
import numpy as np
import nltk
from nltk import ConditionalFreqDist
import nltk.data, nltk.tag
from nltk.corpus import (stopwords, wordnet as wn)
import math
import string

# You might change the window size
# 10:0.524, 11: 0.537,12: 0.534, 12: 0.534, 13:0.535, 14:0.524, 15:0.530
window_size = 10

# B.1.a,b,c,d
def extract_features(data, language='English'):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :return: features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }
    '''
    features = {}
    labels = {}

    # implement your code here
    # bag of words( context window)
    # sense_ids = {}
    # map(lambda instance: sense_ids.setdefault(instance[-1], 0), data)
    # map(lambda instance: sense_ids.update(
        # {instance[-1]: sense_ids[instance[-1]] + 1}),
        # data)
    # senses_count = sum(sense_ids.values())
    def extract_features_each_instance(instance):
        tokens_left = list(nltk.word_tokenize(instance[1]))
        tokens_right = list(nltk.word_tokenize(instance[3]))
        tokens = tokens_left + [instance[2]] + tokens_right
        context_words = tokens_left[-window_size:] + tokens_right[0:window_size]
        feature_window_words = {}

        # b)
        # Remove stop words, remove punctuations, do stemming, etc
        if language.lower() in ['english', 'spanish']:
            # stop words
            stop_words = stopwords.words(language.lower())

            tokens_left = filter(lambda token: token not in stop_words + list(string.punctuation),
                                 list(nltk.wordpunct_tokenize(instance[1])))
            tokens_left = map(lambda token: extract_features.stemmer.stem(token), tokens_left)
            tokens_right = filter(lambda token: token not in stop_words + list(string.punctuation),
                                  list(nltk.wordpunct_tokenize(instance[3])))
            tokens_right = map(lambda token: extract_features.stemmer.stem(token), tokens_right)
            tokens = tokens_left + [instance[2]] + tokens_right
            context_words = tokens_left[-window_size:] + tokens_right[0:window_size]

        def inc_count(word):
            '''
            '''
            key = 'BOW_' + word
            feature_window_words.setdefault(key, 0)
            feature_window_words[key] += 1
        map(inc_count, context_words)

        # a)
        # collocational features
        # surrounding words,-2,...,2.
        # part-of-speech tags,-2,...,2.
        feature_surrounding_words = {}
        feature_surrounding_pos = {}
        collocation_size = 3
        if language.lower() in ['english', 'spanish']:
            if language.lower() == 'spanish':
                collocation_size = 2
            collocation_words = tokens_left[-collocation_size:] +\
                [instance[2]] + tokens_right[0:collocation_size]
            def extract_surrounding_words(index_word):
                key = 'SW_'+unicode(index_word[0])+'_'+index_word[1]
                feature_surrounding_words[key] = 1
            map(extract_surrounding_words, enumerate(collocation_words))

        if language.lower() in ['english', 'spanish']:
            tokens_tagged = extract_features.tagger.tag(tokens)
            tokens_left_tagged = tokens_tagged[0:len(tokens_left)]
            tokens_head_tagged = tokens_tagged[len(tokens_left)]
            tokens_right_tagged = tokens_tagged[len(tokens_left)+1:]
            tokens_surrounding_tagged = tokens_left_tagged[-collocation_size:] \
                + [ tokens_head_tagged] + tokens_right_tagged[0:collocation_size]
            def extract_surrounding_pos(index_word_tag):
                key = 'SPOS_'+unicode(index_word_tag[0] - collocation_size)+'_'+index_word_tag[-1][-1]
                feature_surrounding_pos[key] = 1
            map(extract_surrounding_pos,
                enumerate(tokens_surrounding_tagged))


        # c)
        # relevance score
        feature_relevance_score = {}
        # map(lambda sense_tuple:
            # feature_relevance_score.setdefault('RS_'+unicode(sense_tuple[0]),0), sense_ids)
        # map(lambda sense_tuple: sense_ids.setdefault(
            # 'RS_'+unicode(sense_tuple[0]),
            # math.log(sense_ids[sense_tuple[0]])
            # -math.log(senses_count-sense_ids[sense_tuple[0]])), sense_ids)

        # d)
        # combination of synonyms, hyponyms and hypernyms
        feature_nyms = {}
        def extract_xnyms(index_word):
            synsets = wn.synsets(index_word[1])
            key_pre = unicode(index_word[0])+'_'
            for synset in synsets:
                feature_nyms[key_pre+'SYN_NAME_'+synset.name().split('.')[0]] = 1
                feature_nyms[key_pre+'SYN_POS_'+synset.pos()] = 1
            # for synset in filter(
                # lambda x: x.name().split('.')[0] == index_word[1],synsets):
                for hypernym in synset.hypernyms():
                    feature_nyms[key_pre+'SYN_HYPER_NAME_'+
                                 hypernym.name().split('.')[0]] = 1
                    feature_nyms[key_pre+'SYN_HYPER_POS_'+
                                 hypernym.pos()] = 1
                for hyponym in synset.hyponyms():
                    feature_nyms[key_pre+'SYN_HYPO_NAME_'+
                                 hyponym.name().split('.')[0]] = 1
                    feature_nyms[key_pre+'SYN_HYPO_POS_'+
                                 hyponym.pos()] = 1

        features[instance[0]] = dict(feature_window_words.items()
                                     + feature_surrounding_pos.items()
                                     + feature_surrounding_words.items()
                                     )
        labels[instance[0]] = instance[-1]

    # features = dict(map(extract_features_each_instance, data))
    # labels = dict(map(lambda instance: (instance[0], instance[-1]), data))
    map(extract_features_each_instance, data)

    return features, labels
extract_features.tagger = nltk.tag.PerceptronTagger()
extract_features.tagset = None

# implemented for you
def vectorize(train_features,test_features):
    '''
    convert set of features to vector representation
    :param train_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :param test_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :return: X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
            X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    '''
    X_train = {}
    X_test = {}

    vec = DictVectorizer()
    vec.fit(train_features.values())
    for instance_id in train_features:
        X_train[instance_id] = vec.transform(train_features[instance_id]).toarray()[0]

    for instance_id in test_features:
        X_test[instance_id] = vec.transform(test_features[instance_id]).toarray()[0]

    return X_train, X_test

#B.1.e
def feature_selection(X_train,X_test,y_train):
    '''
    Try to select best features using good feature selection methods (chi-square or PMI)
    or simply you can return train, test if you want to select all features
    :param X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }
    :return:
    '''



    # implement your code here
    X_train_new = X_train
    X_test_new = X_test

    _y_train = map(lambda key: y_train[key], X_train.keys())
    selectorKBest = SelectKBest(chi2, k=len(X_train.items()[0][1]) - 100).fit(
        X_train.values(), _y_train)
    X_train_selected = selectorKBest.transform(X_train.values())
    X_test_selected = selectorKBest.transform(X_test.values())
    X_train_new = dict(map(lambda index_key: (X_train.keys()[index_key[0]], X_train_selected[index_key[0]]),
                        enumerate(X_train.keys())))
    X_test_new = dict(map(lambda index_key: (X_test.keys()[index_key[0]], X_test_selected[index_key[0]]),
                        enumerate(X_test.keys())))
    #return X_train_new, X_test_new
    # or return all feature (no feature selection):
    return X_train_new, X_test_new

# B.2
def classify(X_train, X_test, y_train):
    '''
    Train the best classifier on (X_train, and y_train) then predict X_test labels

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

    :return: results: a list of tuples (instance_id, label) where labels are predicted by the best classifier
    '''

    results = []


    # implement your code here

    # SVM
    svm_clf = svm.LinearSVC(C=1.0, verbose=0, random_state=0)
    knn_clf = neighbors.KNeighborsClassifier(14)
    # the label encoder seems not necessary
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(y_train.values())
    # FATAL & VITAL!!!
    # DictVectorizer `REORDERS` key values in the X_train `DICTIONARY`
    _y_train = map(lambda key: y_train[key], X_train.keys())

    X_train_arr = sparse.csr_matrix(X_train.values())
    X_test_arr = sparse.csr_matrix(X_test.values())
    y_train_arr = np.array(label_encoder.transform(_y_train))
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
    # GBDT

    # ...

    results = svm_results

    return results

# run part B
def run(train, test, language, answer):
    print 'running B for language:', language
    results = {}
    if language.lower() in ['english', 'spanish']:
        extract_features.stemmer = nltk.SnowballStemmer(language.lower())

    for lexelt in train:

        train_features, y_train = extract_features(train[lexelt], language=language)
        test_features, _ = extract_features(test[lexelt], language=language)

        X_train, X_test = vectorize(train_features,test_features)
        if language.lower() in ['english', 'spanish']:
            X_train_new, X_test_new = feature_selection(X_train, X_test,y_train)
        else:
            X_train_new = X_train
            X_test_new = X_test
        results[lexelt] = classify(X_train_new, X_test_new,y_train)

    A.print_results(results, answer)
