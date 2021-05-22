import argparse

import os

import re
import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

import data_loading_saving
import utils

from itertools import groupby

# dowload data
nltk.download('stopwords')
nltk.download('wordnet')

# GLOBAL VARIABLES
STOP_WORDS = set(stopwords.words('english'))
SHORTENED_ABBREVATION_DICT = {
    'a$$': 'ass',
    'a&f': 'always and forever',
     "a'ight": 'alright',
     'a1t': 'anyone there',
     'A2A': 'Ask to Answer',
     'a3': 'anyplace, anywhere, anytime',
     'aak': 'Alive and Kicking',
     'aamof': 'as a matter of fact',
     'abft': 'About fucking Time',
     'aboot': 'about',
     'absnt': 'absent',
     'abt': 'about',
     'acc': 'account',
     'acct': 'account',
     'acgaf': "Absolutely couldn't give a fuck",
     'ack': 'acknowledged',
     'ahole': 'asshole',
     'aight': 'Alright',
     'aiight': 'all right',
     'aka': 'also known as',
     'alot': 'a lot',
     'alrite': 'Alright',
     'alrt': 'alright',
     'alryt': 'alright',
     'any1': 'Anyone',
     'anywaz': 'anyways',
     'apreci8': 'appreciate',
     'asafp': 'as soon as fucking possible',
     'awb': 'Acquaintance with benefits',
     'awsm': 'awesome',
     'awsome': 'awesome',
     'lol': 'laughing out loud',
     'im': 'I am',
     'wanna': 'want to',
     'gona': 'going to',
     'omg': 'oh my god',
     'ok': 'okay',
     'lmao': 'laughing my ass off',
     'soo': 'so',
     'sooo': 'so',
     'gotta': 'got to',
     'idk': 'i do not know',
     'ima': 'i am going to',
     'oomf': 'one of my followers',
     'noo': 'no',
     'nooo': 'no',
     'lil': 'little',
     'rt': 'retweet',
     'bro': 'brother',
     'btw': 'by the way',
     'smh': 'shaking my head',
     'ff': 'surrender',
     'ddr': 'dance dance revolution',
     'kinda': 'kind of',
     'yu': 'you',
     'congrats': 'congratulations',
     'lmfao': 'laughing my ass off',
     'imma': 'i am going to',
     'mr': 'mister',
     'goin': 'going',
     'ive': 'i have',
     'ppl': 'people',
     'pls': 'please',
     'yall': 'you all',
     'lool': 'laughing out loud',
     'loool': 'laughing out loud',
     'ty': 'thank you',
     'thankyou': 'thank you',
     'wtf': 'what the fuck',
     'hun': 'honey',
     'dnt': 'do not',
     '1st': 'first',
     '2nd': 'second',
     'mee': 'me',
     'meee': 'me',
     'dat': 'that',
     'fav': 'favorite',
     'usa': 'united states of america',
     '420': 'marijuana',
     'gettin': 'getting',
     'hes': 'he is',
     'np': 'no problem',
     'outta': 'out of',
     'vs': 'versus',
     'isnt': 'is not',
     '3rd': 'third',
     '4th': 'fourth',
     'bc': 'because',
     'pics': 'pictures',
     'jk': 'just kidding',
     'thanx': 'thank you',
     'thanks': 'thank you',
     'thx': 'thank you',
     'nah': 'no',
     'xoxo': 'hugs and kisses',
     'fml': 'hate my life',
     'goodmorning': 'good morning',
     'tht': 'that',
     'sadtweet': 'sad tweet',
     'tryna': 'trying to',
     'yeah': 'yes',
     'yeahh': 'yes',
     'dc': 'do not care',
     'mins': 'minutes',
     'yess': 'yes',
     'fo': 'fuck off',
     'thru': 'through',
     'boobs': 'breasts',
     'omfg': 'oh my god',
     'doin': 'doing',
     'asap': 'as soon as possible',
     'dunno': 'do not know',
     'yolo': 'you only live once',
     'bf': 'boyfriend',
     'tbh': 'to be honest',
     'convo': 'conversation',
     'ny': 'new york city',
     'nt': 'nice try',
     'tmrw': 'tomorrow',
     'knw': 'know',
     'nw': 'no way',
     'nf': 'not funny',
     'fm': 'fuck me',
     'bff': 'best friend forever',
     'needa': 'need to',
     'yeahhh': 'yes',
     'loveyou': 'love you',
     'yeh': 'yes',
     'wana': 'want to',
     'chillin': 'relaxing',
     'fr': 'for real',
     'hella': 'very',
     'ily': 'i love you',
     'homie': 'good friend',
     'bestie': 'best friend',
     'nba': 'national basketball association',
     'lmfaoo': 'laughin my ass off',
     'lmfaooo': 'laughing my ass off',
     'lmaoo': 'laughing my ass off',
     'lmaooo': 'laughing my ass off',
     'heyy': 'hello',
     'tf': 'what the fuck',
     'nyc': 'new york city',
     'iloveyou': 'i love you',
     'bein': 'being',
     'mb': 'my bad',
     'bestselling': 'best selling',
     'omgg': 'oh my god',
     'omggg': 'oh my god',
     'intel': 'information',
     'comin': 'coming',
     'wassup': 'what is up',
     'whats up': 'what is up',
     'watchin': 'watching',
     'peeps': 'people',
     'b4': 'before',
     'mic': 'microphone',
     'prolly': 'probably',
     'bs': 'bullshit',
     'shoulda': 'should have',
     'yh': 'yeah',
     'dats': 'that is',
     'vip': 'very important person'
}

CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
}


def replace_contraction(tweet: str):
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
        if i in CONTRACTION_MAP:
            result.append(CONTRACTION_MAP.get(i))
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
        if i in SHORTENED_ABBREVATION_DICT:
            result.append(SHORTENED_ABBREVATION_DICT.get(i))
        else:
            result.append(i)
    text = ' '.join(word for word in result)
    return text


def stemming(tweet: str) -> str:
    """Stem a tweet. Stemming is a process of reducing words to their word stem, base or root form (for example, books — book, looked — look).

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


def basic_preprocess(tweet: str, first_pass : bool) -> str:
    """Remove numbers, user, url, hastags, excessive whitespaces tags and force to lower case

    Args:
        tweet (string): tweet as as string

    Returns:
        string: tweet without numbers, user, url, hashtags, excess whitespaces and forced to lower case
    """

    tweet = re.sub('<user>', '', tweet)  # remove user tags
    tweet = re.sub('<url>', '', tweet)  # remove url tags
    tweet = re.sub('[0-9]', '', tweet)  # remove numbers
    if not first_pass : tweet = re.sub('#\w*', '', tweet)  # remove hashtags

    tweet = re.sub('\s+', ' ', tweet)  # remove excess whitespace
    tweet = re.sub('^\s', '', tweet)  # remove excess whitespace
    tweet = re.sub('\s$', '', tweet)  # remove excess whitespace
    tweet = tweet.strip()  # remove star/end whitespaces

    tweet = tweet.lower()  # lower case

    return tweet


def remove_punctuation(tweet: str) -> str:
    """ Remove punct and special chars '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

    Args:
        tweet (string): tweet as string

    Returns:
        string: tweet without punctuation
    """
    tweet = "".join([char for char in tweet if char not in string.punctuation])
    return tweet


def split_hashtags(tweet: str) -> str:
    """ Remove punct and special chars '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

    Args:
        tweet (string): tweet as string

    Returns:
        string: tweet without hashtags and splitted
    """
    hashtags_list = re.findall(r"#(\w+)", tweet)
    splits = [None] * len(hashtags_list)
    for j in range(len(hashtags_list)):
        splits[j] = utils.segment(hashtags_list[j])
        tweet = tweet.replace('#' + str(hashtags_list[j]), ' '.join(splits[j][0]))
    return tweet

def main(args):
    '''
    :args: command line arguments including file paths and verbose flag
    '''
    if args.verbose: print("reading input from %s..." % args.input_path)
    if args.is_test_set:
        df = data_loading_saving.load_raw_data_testset(args.input_path)
    else:
        df = data_loading_saving.load_raw_data_trainingset(args.input_path, args.labels)
    print(len(df))
    if args.verbose: print("basic processing...")
    df['tweet'] = df['tweet'].apply(lambda row: basic_preprocess(str(row), True))
    if args.verbose: print("processing: removing # and splitting hashtag text...")
    df['tweet'] = df['tweet'].apply(lambda row: split_hashtags(str(row)))
    if args.verbose: print("processing: replace contraction...")
    df['tweet'] = df['tweet'].apply(lambda row: replace_contraction(str(row)))

    if args.verbose: print("processing: remove punctuation and special characters...")
    df['tweet'] = df['tweet'].apply(lambda row: remove_punctuation(str(row)))

    if args.abbreviations:
        if args.verbose: print("processing: replace abbreviations...")
        df['tweet'] = df['tweet'].apply(lambda row: replace_abbreviations(str(row)))

    if args.stop_words:
        if args.verbose: print("processing: remove stopwords...")
        df['tweet'] = df['tweet'].apply(lambda row: remove_stop_words(str(row)))

    if args.stemming:
        if args.verbose: print("processing: stemming...")
        df['tweet'] = df['tweet'].apply(lambda row: stemming(str(row)))

    if args.lemmatizing:
        if args.verbose: print("processing: lemmatizing...")
        df['tweet'] = df['tweet'].apply(lambda row: lemmatizing(str(row)))

    if args.verbose: print("basic processing... Again...")
    df['tweet'] = df['tweet'].apply(lambda row: basic_preprocess(str(row), False))

    if args.verbose: print("writing output to %s..." % args.output_path)
    dir_out = os.path.dirname(args.output_path)
    if dir_out != "" and os.path.exists(dir_out):
        data_loading_saving.write_to_text(dir_out +'/'+ args.output_file, df)

    #dir_out = os.path.dirname(args.output_path)
    #if dir_out != "" and os.path.exists(dir_out):
    data_loading_saving.write_to_text(args.output_path, df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='takes raw data, outputs preprocessed data')
    parser.add_argument('input_path', type=str, help='path to raw data', action='store')
    parser.add_argument('output_path', type=str, help='path where output should be written', action='store')
    parser.add_argument('output_file', type=str, help='file where output should be written', action='store')
    parser.add_argument('is_test_set', type=bool, help='do you preprocess testing set ?', action='store')
    parser.add_argument('labels', type=int, help='0 for negative, 1 for positive, anything else if is_test_set',
                        action='store')
    parser.add_argument('abbreviations', type=bool, help='do you want to abbreviate tweet', action='store')
    parser.add_argument('stemming', type=bool, help='do you want to stemm tweet', action='store')
    parser.add_argument('lemmatizing', type=bool, help='do you want to lemmatize tweet ?', action='store')
    parser.add_argument('stop_words', type=bool, help='do you to remove stop words?', action='store')
    parser.add_argument('-v', '--verbose', dest='verbose', help='want verbose output or not?', action='store_true')
    args = parser.parse_args()
    main(args)
