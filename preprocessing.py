import re
import sys

#import wordsegment as ws
#ws.load()
#import autocorrect as ac
#speller = ac.Speller(lang='en')

VERBOSE = 1

def preprocess_tweet(tweet):
    '''
    :param tweet: tweet as a string
    :return: string s which is preprocessed tweet
    '''
    tweet = re.sub('<user>', '', tweet) # remove user tags
    tweet = re.sub('<url>', '', tweet) # remove url tags
    tweet = re.sub('#\w*', '', tweet) # remove hashtags
    tweet = re.sub('[0-9]', '', tweet) # remove numbers
    tweet = re.sub('[^\w\s]', '', tweet) # remove punctuation
    tweet = re.sub('\s+', ' ', tweet) # remove excess whitespace
    tweet = re.sub('^\s', '', tweet) # remove excess whitespace
    tweet = re.sub('\s$', '', tweet) # remove excess whitespace
    tweet += '\n'
    #tweet = ' '.join(ws.segment(tweet)) # segment words
    #tweet = ' '.join([speller(w) for w in tweet.split()]) # spell checking
    return tweet

def main(PATH_INPUT, PATH_OUTPUT):
    if VERBOSE:
        print("reading input from %s..." % PATH_INPUT)    
    f_in = open(PATH_INPUT, "r")
    l = f_in.readlines()
    
    if VERBOSE:
        print("processing...")    
    for i in range(len(l)):
        if i % 1000 == 0:
            print("%d of %d" % (i, len(l)))
        l[i] = preprocess_tweet(l[i])
    
    if VERBOSE:
        print("writing output to %s..." % PATH_OUTPUT)    
    f_out = open(PATH_OUTPUT, "w")
    f_out.writelines(l)

if __name__ == "__main__":
    if(len(sys.argv) < 3):
        print("arg1: input path, arg2: output path")
    else:
        main(sys.argv[1], sys.argv[2])

