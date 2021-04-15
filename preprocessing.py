import re
import wordsegment as ws
ws.load()
import autocorrect as ac
speller = ac.Speller(lang='en')
#import wordsegmentation as ws
#segmenter = ws.WordSegment()

def preprocess_tweet(tweet):
    '''
    :param tweet: tweet as a string
    :return: string s which is preprocessed tweet
    '''
    tweet = re.sub('<user>', '', tweet)
    tweet = re.sub('<url>', '', tweet)
    tweet = re.sub('[^\w\s]', '', tweet) # punctuation
    tweet = re.sub('[0-9]', '', tweet) # numbers
    #tweet = re.sub('USER', '<user>', tweet)
    #tweet = re.sub('URL', '<url>', tweet)
    #tweet = ' '.join([segmenter.segment(tweet) for w in tweet.split()])
    return tweet

def spellcheck_tweet(tweet):
    tweet = ' '.join([speller(w) for w in tweet.split()])
    tweet = ' '.join(ws.segment(tweet))
