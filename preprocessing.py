import re
import sys
import argparse

#import wordsegment as ws
#ws.load()
#import autocorrect as ac
#speller = ac.Speller(lang='en')

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

def main(args):
    '''
    :args: command line arguments including file paths and verbose flag
    '''
    if args.verbose: print("reading input from %s..." % args.input_path)    
    f_in = open(args.input_path, "r")
    l = f_in.readlines()
    
    if args.verbose: print("processing...")    
    for i in range(len(l)):
        if i % 1000 == 0:
            if args.verbose: print("%d of %d" % (i, len(l)))
        l[i] = preprocess_tweet(l[i])
    
    if args.verbose: print("writing output to %s..." % args.output_path)    
    f_out = open(args.output_path, "w")
    f_out.writelines(l)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='takes raw data, outputs preprocessed data')
    parser.add_argument('input_path', type=str, help='path to raw data', action='store')
    parser.add_argument('output_path', type=str, help='path where output should be written', action='store')
    parser.add_argument('-v', '-verbose', dest='verbose', help='want verbose output or not?', action='store_true')
    args = parser.parse_args()
    main(args)

