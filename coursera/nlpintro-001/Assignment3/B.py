import A
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn import preprocessing
from scipy import sparse
import numpy as np
import nltk


# You might change the window size
window_size = 10

# B.1.a,b,c,d
def extract_features(data):
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
    word_bag = A.build_s({'lexelt': data})['lexelt']
    def extract_features_each_instance(instance):
        tokens_left = list(nltk.word_tokenize(instance[1]))
        tokens_right = list(nltk.word_tokenize(instance[3]))
        tokens = tokens_left + [instance[2]] + tokens_right
        feature_window_words = {}
        context_words = tokens_left[-window_size:] + tokens_right[0:window_size]

        # b)
        # Remove stop words, remove punctuations, do stemming, etc

        def extract_bag_feature(word):
            '''
            '''
            feature_window_words.setdefault('BOW_'+word, 0)
            feature_window_words['BOW_'+word] += 1
        map(extract_bag_feature, context_words)

        # a)
        # collocational features
        # surrounding words,-2,...,2.
        # part-of-speech tags,-2,...,2.
        feature_surrounding_words = {}
        feature_surrounding_pos = {}
        collocation_size = 2
        collocation_words = tokens_left[-collocation_size:] +\
            [instance[2]] + tokens_right[0:collocation_size]
        def extract_surrounding_words(index_word):
            try:
                feature_surrounding_words.setdefault('SW_'+unicode(index_word[0])+'_'+index_word[1], 0)
                feature_surrounding_words['SW_'+unicode(index_word[0])+'_'+index_word[1]] += 1
            except Exception as e:
                print( e, collocation_words)
                pass
        map(extract_surrounding_words, enumerate(collocation_words))
        # nltk.tag.HiddenMarkovModelTagger()
        # tagger = nltk.tag.StanfordPOSTagger()
        # print('tagging tokens')

        # tokens_tagged = nltk.pos_tag(tokens)
        # tokens_left_tagged = tokens_tagged[0:len(tokens_left)]
        # tokens_head_tagged = tokens_tagged[len(tokens_left)]
        # tokens_right_tagged = tokens_tagged[len(tokens_left)+1:]
        # def extract_surrounding_pos(index_word_tag):
            # feature_surrounding_pos.setdefault('SPOS_'+unicode(index_word_tag[0])+'_'+index_word_tag[-1], 0)
            # feature_surrounding_pos['SPOS_'+unicode(index_word_tag[0])+'_'+index_word_tag[-1]] += 1
        # map(extract_surrounding_pos, enumerate(tokens_left_tagged +[tokens_head_tagged]+ tokens_right_tagged))


        # c)
        # relevance score

        # d)
        # combination of synonyms, hyponyms and hypernyms

        # e)
        # FEATURE SELECTION:Chi-square or pointwise mutual information (PMI)
        # see function feature_selection() below
        return (instance[0], dict(map(lambda x: x,
                                      feature_window_words.items() +
                                      feature_surrounding_pos.items() +
                                      feature_surrounding_words.items()
                                      )))

    features = dict(map(extract_features_each_instance, data))
    labels = dict(map(lambda instance: (instance[0], instance[-1]), data))

    return features, labels

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

    #return X_train_new, X_test_new
    # or return all feature (no feature selection):
    return X_train, X_test

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
    svm_clf = svm.LinearSVC(C=1.01)
    # the label encoder seems not necessary
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(y_train.values())

    X_train_arr = sparse.csr_matrix(np.array(X_train.values()))
    X_test_arr = sparse.csr_matrix(np.array(X_test.values()))
    y_train_arr = np.array(label_encoder.transform(y_train.values()))
    # X_test_arr = np.array(X_test.values())
    svm_clf.fit(X_train_arr, y_train_arr)
    svm_predicted = svm_clf.predict(X_test_arr)
    svm_results = zip(X_test.keys(),
                      label_encoder.inverse_transform(svm_predicted))
    # GBDT

    # ...

    results = svm_results

    return results

# run part B
def run(train, test, language, answer):
    results = {}

    for lexelt in train:

        train_features, y_train = extract_features(train[lexelt])
        test_features, _ = extract_features(test[lexelt])

        X_train, X_test = vectorize(train_features,test_features)
        X_train_new, X_test_new = feature_selection(X_train, X_test,y_train)
        results[lexelt] = classify(X_train_new, X_test_new,y_train)

    A.print_results(results, answer)
