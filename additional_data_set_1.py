import pandas as pd
import numpy as np 
import re
import argparse

# The dataset that you should use with this script can be found here: https://www.kaggle.com/c/tweet-sentiment-extraction/data?select=train.csv
# It is from a kaggle competition "Tweet Sentiment Extraction"
# We keep only pos/neg tweets (not neutral) 

#GLOBAL VARIABLES
output_path_pos = "train_pos_add1.txt"
output_path_neg = "train_neg_add1.txt"

def read_dataset(path: str) -> pd.DataFrame:
    """read the dataset and store it in a DataFrame

    Args:
        path (str): dataset from https://www.kaggle.com/c/tweet-sentiment-extraction/data?select=train.csv
    """
    data = pd.read_csv(path)
    return data


def format_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """give the correct format to the dataframe (keep only useful columns) and keep only pos/neg tweets

    Args:
        data (pd.DataFrame): The dataframe to be formated

    Returns:
        pd.DataFrame: a dataframe containting tweets and sentiment (pos/neg only) 
    """
    data = data[['text', 'sentiment']]
    return data.loc[data['sentiment'] != "neutral"]

def pre_processing(tweet: str) -> str:
    """Remove user and url as in the official datasets

    Args:
        tweet (str): a tweet (text) to preprocess

    Returns:
        str: the tweet without user and url 
    """
    tweet = re.sub('@[^\s]+','<user>',tweet)
    tweet = re.sub(r'http\S+', '<url>', tweet)
    return tweet

def create_datasets(data: pd.DataFrame):
    """create 2 training datasets (for pos and neg tweets) in the same format than the official ones

    Args:
        data (pd.DataFrame): dataframe of 2 columns (tweet and sentiment) used to create the 2 training datasets
    """
    
    positive_tweet = data.loc[data['sentiment'] == "positive"][['text']]
    negative_tweet = data.loc[data['sentiment'] == "negative"][['text']]
    
    positive_tweet.to_csv("Data/" + output_path_pos, header=None, index=None, sep=',', mode='a',)
    negative_tweet.to_csv("Data/" + output_path_neg, header=None, index=None, sep=',', mode='a',)

    return

def main(args):
    dataFrame = read_dataset(args.input_path)
    dataFrame = format_dataset(dataFrame)
    dataFrame['text'] = dataFrame['text'].apply(lambda row: pre_processing(str(row)))
    dataFrame
    create_datasets(dataFrame)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='takes additional dataset, outputs the same dataset in the correct format and with a pos and neg .txt files')
    parser.add_argument('input_path', type=str, help='path to raw data', action='store')
    args = parser.parse_args()
    main(args)

