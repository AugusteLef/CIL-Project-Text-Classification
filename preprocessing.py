# packages
import argparse
import re
import string
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

# import custom utils
import utils

# load wordlists into global variables
STOP_WORDS = set(stopwords.words('english'))
with open('Preprocessing_Data/contractions.json', 'r') as file:
    CONTRACTIONS = json.load(file)
with open('Preprocessing_Data/abbreviations.json', 'r') as file:
    ABBREVIATIONS = json.load(file)

def replace_contractions(tweet: str):
    """ Replace contraction in a text
    
    Args:
        tweet (string): tweet as a string
    
    Returns:
        string: tweet with contractions removed
    """
    tknzr = TweetTokenizer()
    words = tknzr.tokenize(tweet)
    result = []
    for i in words:
        if i in CONTRACTIONS:
            result.append(CONTRACTIONS.get(i))
        else:
            result.append(i)
    text = ' '.join(word for word in result)
    return text

def replace_abbreviations(tweet: str):
    """ Replace contraction in a text

    Args:
        tweet (string): tweet as a string

    Returns:
        string: tweet with contractions removed
    """
    tknzr = TweetTokenizer()
    words = tknzr.tokenize(tweet)
    result = []
    for i in words:
        if i in ABBREVIATIONS:
            result.append(ABBREVIATIONS.get(i))
        else:
            result.append(i)
    text = ' '.join(word for word in result)
    return text

def stemming(tweet: str) -> str:
    """ Stem a tweet. Stemming is a process of reducing words to their word stem, base or root form (for example, books — book, looked — look).

    Args:
        tweet (string): tweet as a string

    Returns:
        string: tweet stemmed
    """
    stemmer = PorterStemmer()
    tokens = word_tokenize(tweet)
    result = [stemmer.stem(i) for i in tokens]
    text = ' '.join(word for word in result)
    return text

def lemmatizing(tweet: str) -> str:
    """Lemmatize a tweet. The aim of lemmatization, like stemming, is to reduce inflectional forms to a common base form.
    As opposed to stemming, lemmatization does not simply chop off inflections.
    Instead it uses lexical knowledge bases to get the correct base forms of words. [ex: running - run]

    Args:
        tweet (string): tweet as a string

    Returns:
        string: tweet lemmatized
    """
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(tweet)
    result = [lemmatizer.lemmatize(i) for i in tokens]
    text = ' '.join(word for word in result)
    return text

def remove_stop_words(tweet: str) -> str:
    """ Remove stop words. A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) that dont give any
    information (for sentiment analysis) and takes space in the database and increases the processing time.

    Args:
        tweet (string): tweet as a string

    Returns:
        string: tweet without stop words
    """
    tokens = word_tokenize(tweet)
    result = [i for i in tokens if not i in STOP_WORDS]
    text = ' '.join(word for word in result)
    return text

def basic_preprocess(tweet: str) -> str:
    """Remove numbers, user, url, hastags, excessive whitespaces tags and force to lower case

    Args:
        tweet (string): tweet as as string

    Returns:
        string: tweet without numbers, user, url, hashtags, excess whitespaces and forced to lower case
    """
    tweet = re.sub('<user>', '', tweet)  # remove user tags
    tweet = re.sub('<url>', '', tweet)  # remove url tags
    tweet = re.sub('[0-9]', '', tweet)  # remove numbers
    tweet = re.sub('#\w*', '', tweet)  # remove hashtags
    tweet = re.sub('\s+', ' ', tweet)  # remove excess whitespace
    tweet = re.sub('^\s', '', tweet)  # remove excess whitespace
    tweet = re.sub('\s$', '', tweet)  # remove excess whitespace
    tweet = tweet.strip()  # remove star/end whitespaces
    tweet = tweet.lower()  # lower case
    return tweet

def remove_punctuation(tweet: str) -> str:
    """ Remove punctuation and special chars '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

    Args:
        tweet (string): tweet as string

    Returns:
        string: tweet without punctuation
    """
    tweet = "".join([char for char in tweet if char not in string.punctuation])
    return tweet

def data_augmentation(tweet: str) -> str:
    """ Creates a new tweet replacing a word by its synonym using the Thesaurus-based subsititution

    Args:
        tweet (string): tweet as string

    Returns:
        string: new augmented tweet
    """

    return None

def main(args):
    """ runs the whole preprocessing pipeline according to command line arguments

    Args: 
        command line arguments including file paths and verbose flag
    """   
    # reading data
    if args.verbose: print("reading input from %s..." % args.input_path)
    df = utils.load_raw_data(args.input_path)
    print("number of tweets: %s" % len(df))   
    # basic preprocessing
    if args.verbose: print("basic processing...")
    df['tweet'] = df['tweet'].apply(lambda row: basic_preprocess(str(row)))
    # contractions
    if args.verbose: print("processing: replace contraction...")
    df['tweet'] = df['tweet'].apply(lambda row: replace_contractions(str(row)))
    # abbreviations
    if args.verbose: print("processing: replace abbreviations...")
    df['tweet'] = df['tweet'].apply(lambda row: replace_abbreviations(str(row)))
    # punctuation
    if args.verbose: print("processing: remove punctuation and special characters...")
    df['tweet'] = df['tweet'].apply(lambda row: remove_punctuation(str(row)))
    # stop words
    if args.augmentation:
        if args.verbose: print("processing: data augmentation...")
        augmentation = df['tweet'].apply(lambda row: data_augmentation((str(row))))
        #add the augmentation to df...
    if args.stop_words:
        if args.verbose: print("processing: remove stopwords...")
        df['tweet'] = df['tweet'].apply(lambda row: remove_stop_words(str(row)))
    # stemming
    if args.stemming:
        if args.verbose: print("processing: stemming...")
        df['tweet'] = df['tweet'].apply(lambda row: stemming(str(row)))
    # lemmatizing
    if args.lemmatizing:
        if args.verbose: print("processing: lemmatizing...")
        df['tweet'] = df['tweet'].apply(lambda row: lemmatizing(str(row)))
    # basic preprocessing
    if args.verbose: print("basic processing... Again...")
    df['tweet'] = df['tweet'].apply(lambda row: basic_preprocess(str(row)))
    # writing output
    if args.verbose: print("writing output to %s..." % args.output_path)
    df.to_csv(args.output_path, header=False, index=None, sep=',')

# running this file from command-line will do a full preprocessing pass on specified data
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='takes raw data, outputs preprocessed data')
    parser.add_argument('input_path', type=str, help='path to raw data', action='store')
    parser.add_argument('output_path', type=str, help='path where output should be written', action='store')
    parser.add_argument('-s', '--stemming', dest='stemming', help='do you want to stemm tweet?', action='store_true')
    parser.add_argument('-l', '--lemmatizing', dest='lemmatizing', help='do you want to lemmatize tweet?', action='store_true')
    parser.add_argument('-sw', '--stop_words', dest='stop_words', help='do you to remove stop words?', action='store_true')
    parser.add_argument('-a', '--augmentation', dest='augmentation', help='want to do data augmentation?', action='store_true')
    parser.add_argument('-v', '--verbose', dest='verbose', help='want verbose output or not?', action='store_true')
    args = parser.parse_args()
    main(args)
