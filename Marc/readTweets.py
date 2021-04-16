'''
THOUGHTS TO MAYBE RESEARCH:
- add some kind of a bayesian multiplier at the end of the +/- prediction with some extracted feature
    - for instance, when we see multiple repetition of letters (like wowwww,  soooo) --> excitement

'''
import re
import requests
import json
import bs4


def read_tweets(path):
    with open(path) as f:
        tweets = f.readlines()
    f.close
    return tweets


def preprocess(tweet):
    to_remove = ['<users>', '<url>', '#', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[^\w\s]']
    to_deabraviate = {'omg':'oh my god', 'idk':'i do not know', 'dunno': 'do not know' }
    better_tweet = tweet.lower()
    better_tweet = deabreviate(better_tweet, to_deabraviate)
    better_tweet = remove_useless_characters(better_tweet, to_remove)
    return better_tweet


def remove_useless_characters(tweet, to_remove):
    for word in to_remove:
        tweet = re.sub(word, ' ', tweet, flags=re.IGNORECASE)
    return re.sub(' +', ' ', tweet)



def deabreviate(tweet, to_deabraviate):
    #for abrevation, full_word in to_deabraviate.items():
    #    tweet = re.sub(abrevation, full_word, tweet, flags=re.IGNORECASE)
    print(tweet + 'here')
    with open('myslang.json', 'r') as f:
        dict_slang = json.load(f)
    for abbreviation, full_word in dict_slang.items():
        if ('*' not in abbreviation and '?' not in abbreviation and '\M/' not in abbreviation):
            tweet = re.sub(abbreviation, full_word, tweet, flags=re.IGNORECASE)
    print("hihi"+ tweet + 'here')

    return tweet


#one time use
def get_slang_dict():
    resp = requests.get("http://www.netlingo.com/acronyms.php")
    soup = bs4.BeautifulSoup(resp.text, "html.parser")
    slangdict = {}
    key = ""
    value = ""
    for div in soup.findAll('div', attrs={'class': 'list_box3'}):
        for li in div.findAll('li'):
            for a in li.findAll('a'):
                key = a.text
                value = li.text.split(key)[1]
                slangdict[key] = value

    with open('myslang.json', 'w') as f:
        json.dump(slangdict, f, indent=2)