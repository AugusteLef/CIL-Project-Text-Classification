import os
import sys
import pandas as pd
import numpy as np


def load_raw_data_trainingset(path: str, label: int) -> pd.DataFrame:
    """Create a Dataframe containing each tweet and its label

    Args:
        path (str): The path of the file (.txt) to load tweets from
        label (int): The label of tweets (1 for positive, 0 for negative)

    Returns:
        DataFrame: a Dataframe with each tweet from path and its label
    """

    data = []
    with open(path) as file:
        for line in file:
            data.append(line)
    data_df = pd.DataFrame(data, columns = {'tweet'})
    data_df['label'] = label
    return data_df

def load_raw_data_testset(path: str) -> pd.DataFrame:
    """Create a Dataframe containing each tweet

    Args:
        path (str): The path of the file (.txt) to load tweets from

    Returns:
        DataFrame: a Dataframe with each tweet from path
    """

    data = []
    with open(path) as file:
        for line in file:
            data.append(line)
    data_df = pd.DataFrame(data, columns = {'tweet'})
    return data_df


def load_dataframe(path: str) -> pd.DataFrame:
    """Load a dataframe containing tweet and labels

    Args:
        path (str): path to the dataframe (.txt)

    Returns:
        DataFrame: the loaded dataframe
    """
    df = pd.read_csv(path, sep = ',')
    return df

def concat_DataFrame(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Concatenate 2 Dataframes

    Args:
        df1 (DataFrame): first data frame
        df2 (DataFrame): 2nd data frame

    Returns:
        DataFrame: concatenation of dset1 and dset2
    """
    df = pd.concat([df1, df2])
    return df

def write_to_text(path: str, df: pd.DataFrame) -> None:
    """write a pandas datafram txt format in the correct path

    Args:
        path (str): path to write
        df (DataFrame): dataframe to save
    """
    df.to_csv(path, header=True, index=None, sep=',')
    return

def shuffle_df(df: pd.DataFrame) -> pd.DataFrame:
    """shuffle a DataFrame

    Args:
        df (DataFrame): DataFram to shuffle

    Returns:
        DataFrame: DataFrame shuffled
    """
    return df.sample(frac=1)







