import pandas as pd
import numpy as np 
import re
import argparse

# The dataset that you should use with this script can be find here:https://www.kaggle.com/kazanova/sentiment140?select=training.1600000.processed.noemoticon.csv
# It is from a kaggle competition "Sentiment140 dataset with 1.6 million tweets"

#GLOBAL VARIABLES
output_path_pos = "train_pos_add2.txt"
output_path_neg = "train_neg_add2.txt"

def read_dataset(path: str) -> pd.DataFrame:
    """read the dataset and store it in a DataFrame

    Args:
        path (str): dataset from https://www.kaggle.com/c/tweet-sentiment-extraction/data?select=train.csv
    """
    data = pd.read_csv(path, encoding = "ISO-8859-1", header=None)
    return data


def format_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """give the correct format to the dataframe (keep only useful columns)

    Args:
        data (pd.DataFrame): The dataframe to be formated

    Returns:
        pd.DataFrame: a dataframe with the correct format (tweet and sentiment)
    """
    data = data[[0, 5]]
    col_list = list(data)
    col_list[0], col_list[1] = col_list[1], col_list[0]
    data.columns = col_list
    data = data.rename(columns={data.columns[0]: "sentiment", data.columns[1]: "text"})
    data = data[['text','sentiment']]
    
    return data

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
    
    positive_tweet = data.loc[data['sentiment'] == 4][['text']]
    negative_tweet = data.loc[data['sentiment'] == 0][['text']]
    
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

