import torch
import pandas as pd

def load_raw_data(path: str) -> pd.DataFrame:
    """Create a Dataframe containing each tweet

    Args:
        path (str): The path of the file (.txt) to load tweets from

    Returns:
        DataFrame: a Dataframe with each tweet from path and its label
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

def shuffle_df(df: pd.DataFrame) -> pd.DataFrame:
    """shuffle a DataFrame

    Args:
        df (DataFrame): DataFram to shuffle

    Returns:
        DataFrame: DataFrame shuffled
    """
    return df.sample(frac=1)
    
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.labels != None:
            return self.texts[idx], self.labels[idx]
        else:
            return (self.texts[idx],)

class TextCollator():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, list_items):
        texts = [item[0] for item in list_items]
        batch = self.tokenizer(texts, truncation=True, padding=True)
        if 1 < len(list_items[0]):
            labels = [item[1] for item in list_items]
            batch["labels"] = labels
        batch = {key: torch.tensor(val) for key, val in batch.items()}
        return batch

