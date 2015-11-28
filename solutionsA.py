import math
import nltk
import time
import sys

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are
# tuples expressing the ngram and the value is the log probability of that
# ngram


def calc_probabilities(training_corpus):
    """
        this is docstring
    """
    # unigram_tuples = []
    # bigram_tuples = []
    # trigram_tuples = []

    unigram_count = {}
    bigram_count = {}
    trigram_count = {}

    unigram_count_pnodes = {}
    bigram_count_pnodes = {}
    trigram_count_pnodes = {}

    unigram_total = 0
    bigram_total = 0
    trigram_total = 0
    print 'total {} sentences'.format(len(training_corpus))
    for i in xrange(0, len(training_corpus)):
        if i % 3000 == 0:
            print 'processing ', i, 'th sentence...'
        training_corpus[i] = START_SYMBOL + ' ' + training_corpus[i]
        training_corpus[i] = training_corpus[i] + ' ' + STOP_SYMBOL
        # training_corpus[i].replace('.',' ' + STOP_SYMBOL)
        tokens = training_corpus[i].split()
        unigram_tuples_i = list((token,) for token in tokens)
        bigram_tuples_i = list(nltk.bigrams(tokens))
        trigram_tuples_i = list(nltk.trigrams(tokens))
        unigram_total += len(unigram_tuples_i)
        bigram_total += len(bigram_tuples_i)
        trigram_total += len(trigram_tuples_i)
        for item in unigram_tuples_i:
            unigram_count.setdefault(item, 0)
            unigram_count_pnodes.setdefault(item[0:-1], 0)
            unigram_count[item] = unigram_count[item] + 1
            unigram_count[item] = unigram_count_pnodes[item[0:-1]] + 1
        for item in bigram_tuples_i:
            bigram_count.setdefault(item, 0)
            bigram_count_pnodes.setdefault(item[0:-1], 0)
            bigram_count[item] = bigram_count[item] + 1
            bigram_count_pnodes[item[0:-1]] = bigram_count_pnodes[item[0:-1]]+1
        for item in trigram_tuples_i:
            trigram_count.setdefault(item, 0)
            trigram_count_pnodes.setdefault(item[0:-1], 0)
            trigram_count[item] = trigram_count[item] + 1
            trigram_count_pnodes[item[0:-1]] = trigram_count_pnodes[item[0:-1]]+1
    unigram_p = {
        item: math.log(unigram_count[item], 2) - math.log(unigram_total, 2)
        for item in set(unigram_count)}
    bigram_p = {
        item: math.log(bigram_count[item], 2) - math.log(bigram_count_pnodes[item[0:-1]], 2)
        for item in set(bigram_count)}
    trigram_p = {
        item: math.log(trigram_count[item], 2) - math.log(trigram_count_pnodes[item[0:-1]], 2)
        for item in set(trigram_count)}
    print "calc_probabilities finished!"
    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the
# ngram, and the value is the log probability of that ngram


def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = sorted(unigrams.keys())
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' +
                      unigram[0] +
                      ' ' +
                      str(unigrams[unigram]) +
                      '\n')
        outfile.flush()

    bigrams_keys = sorted(bigrams.keys())
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' +
                      bigram[0] +
                      ' ' +
                      bigram[1] +
                      ' ' +
                      str(bigrams[bigram]) +
                      '\n')
        outfile.flush()

    trigrams_keys = sorted(trigrams.keys())
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' +
                      trigram[0] +
                      ' ' +
                      trigram[1] +
                      ' ' +
                      trigram[2] +
                      ' ' +
                      str(trigrams[trigram]) +
                      '\n')
        outfile.flush()

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first
# element is the score of the first sentence, etc.
def score(ngram_p, n, corpus):
    print "scoring corpus for ", n, "-grams"
    scores = []
    conditional_probabilities = {}
    ngram_p_keys = ngram_p.keys()
    for i, sentence in enumerate(corpus):
        score_i = 0
        if i % 100 == 0:
            print 'scoring ', i, 'th sentence...'
        tokens = sentence.split()
        if n == 1:
            ngram_tuples = list([(token,) for token in tokens])
            try:
                score_i = sum([ngram_p[gram] for gram in ngram_tuples])
            except KeyError as error:
                score_i = MINUS_INFINITY_SENTENCE_LOG_PROB
                print 'ngram_tuple ', gram, ' not in dict ', error.message
        elif n == 2 or n == 3:
            ngram_tuples = None
            if n == 2:
                ngram_tuples = list(nltk.bigrams(tokens))
            if n == 3:
                ngram_tuples = list(nltk.trigrams(tokens))
            try:
                for gram in ngram_tuples:
                    conditional_probabilities[gram] = ngram_p[gram] - math.log(
                        sum([math.pow(2, ngram_p[tokens])
                             for tokens in ngram_p_keys if gram[0:-1] == tokens[0:-1]]), 2)
                    score_i += conditional_probabilities[gram]
            except KeyError as error:
                score_i = MINUS_INFINITY_SENTENCE_LOG_PROB
                print 'ngram_tuple not in dict ', error.message
        scores.append(score_i)
    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name


def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores


def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []
    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION


def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')
    sys.exit(0)

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close()

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print("Part A time: " + str(time.clock()) + ' sec')

if __name__ == "__main__":
    main()
