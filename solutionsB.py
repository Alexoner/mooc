import sys
import nltk
import math
import time
import re
import numpy as np

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in your returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []
    for i, sentence_tagged in enumerate(brown_train):
        brown_words_i = []
        brown_tags_i = []
        words_tagged = ' '.join([START_SYMBOL+'/'+START_SYMBOL,
                                 START_SYMBOL+'/'+START_SYMBOL,
                                 sentence_tagged,
                                 STOP_SYMBOL+'/'+STOP_SYMBOL,
                                ]).split()
        re_pattern = re.compile(r'(.*)/([^0-9]+)')
        for word_tagged in words_tagged:
            matched = re_pattern.match(word_tagged)
            brown_words_i.append(matched.group(1))
            brown_tags_i.append(matched.group(2))
        brown_words.append(brown_words_i)
        brown_tags.append(brown_tags_i)

    return brown_words, brown_tags


# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}
    q_count = {}
    q_count_pnodes = {}
    for sentence_tag in brown_tags:
        tags = sentence_tag
        trigram_tuples = nltk.trigrams(tags)
        for gram in trigram_tuples:
            q_count.setdefault(gram, 0)
            q_count_pnodes.setdefault(gram[0:-1], 0)
            q_count[gram] = q_count[gram] + 1
            q_count_pnodes[gram[0:-1]] = q_count_pnodes[gram[0:-1]] + 1
    q_values = {gram: math.log(q_count[gram], 2) - math.log(q_count_pnodes[gram[0:-1]], 2)
                for gram in set(q_count)}
    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    known_words = set([])
    word_dict = {}
    for words in brown_words:
        for word in words:
            word_dict.setdefault(word, 0)
            word_dict[word] = word_dict[word] + 1
            if word_dict[word] > RARE_WORD_MAX_FREQ:
                known_words.add(word)
    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []
    for words in brown_words:
        for i, word in enumerate(words):
            if word not in known_words:
                words[i] = RARE_SYMBOL

        brown_words_rare.append(words)
    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    e_values = {}
    taglist = set([])
    tags_words_count = {}
    tags_count = {}
    for i, words_rare in enumerate(brown_words_rare):
        for j, word in enumerate(words_rare):
            tags_words_count.setdefault((word, brown_tags[i][j]), 0)
            tags_words_count[(word, brown_tags[i][j])] = tags_words_count[
                (word, brown_tags[i][j])] + 1
            tags_count.setdefault(brown_tags[i][j], 0)
            tags_count[brown_tags[i][j]] = tags_count[brown_tags[i][j]] + 1
            taglist.add(brown_tags[i][j])

    e_values = {(word, tag) : math.log(cnt, 2) -
                              math.log(tags_count[tag], 2)
                for (word, tag), cnt in tags_words_count.items()}
    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []
    for i, dev_words in enumerate(brown_dev_words):
        if i+1 % 100 == 0:
            print "tagging",i+1,"th sentence"
        tokens = [START_SYMBOL]*2 + dev_words.split() + [STOP_SYMBOL]
        # tokens = dev_words.split()
        tagged_i = viterbi_dp(tokens, taglist, known_words, q_values, e_values)
        tagged.append(tagged_i)
    import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
    return tagged

def viterbi_dp(tokens, tagset, known_words, q_values, e_values):
    """
    Dynamic programming routine in viterbi algorithm
    """
    message = [{}, {}]
    backpointer = [{}, {}]
    tags = list(tagset)
    wordlist = list(known_words) + [RARE_SYMBOL]
    K = len(tags)
    N = len(tokens)

    # initialization
    for i in xrange(K):
        for j in xrange(K):
            message[1][tags[i], tags[j]] = LOG_PROB_OF_ZERO
    message[1][START_SYMBOL, START_SYMBOL] = 0

    #maintenance
    for n in xrange(2, N-1):
        message.append({})
        backpointer.append({})

        for j in xrange(K):
            for k in xrange(K):
                message_max = float('-inf')
                for i in xrange(K):
                    if tokens[n] not in known_words:
                        print "rare word", tokens[n]
                        token = RARE_SYMBOL
                    else:
                        token = token[n]
                    # consider only emission probability greater than zero
                    if (token, tags[k]) not in e_values:
                        continue
                    message[n][j, k] = max(message[n-1][i, j] +
                                           q_values[tags[i], tags[j], tags[k]] +
                                           e_values[tags[k], token])
                    if message[n][j, k] > message_max:
                        message_max = message[n][j, k]
                        backpointer[n][j, k] = i

    # termination
    path = [-1] * N
    (path[n-3], path[n-2]) = message[N-1].keys()[np.argmax(message[N-1].values())]

    for n in xrange(n-4, 1, -1):
        path[n] = backpointer[n+2][path[n+1], path[n+2]]

    # collect result
    return

# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a
# terminal newline, not a list of tokens.
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i],brown_tags[i]) for i in xrange(len(brown_words)) ]

    # IMPLEMENT THE REST OF THE FUNCTION HERE
    tagged = []
    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare
    sys.exit(0)

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
