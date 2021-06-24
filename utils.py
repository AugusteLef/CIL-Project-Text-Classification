import torch
import pandas as pd

def load_raw_data(path: str) -> pd.DataFrame:
    """Create a Dataframe containing each tweet

    Args:
        path (str): The path of the file (.txt) to load tweets from

    Returns:
        DataFrame: a Dataframe with one row per tweet
    """
    data = []
    with open(path) as file:
        for line in file:
            data.append(line)
    data_df = pd.DataFrame(data, columns = {'tweet'})
    return data_df
    
#
def XLNET_tweets_transformation(df):
    """ XLNET need sep and cls tags at the end of each tweet

    Args:
        list of tweets

    Returns: 
        list of processed tweets
    """
    tweets = []
    for tweet in df['tweet']:
        tweet = tweet+'[SEP] [CLS]'
        tweets.append(tweet)
    return tweets

class TextDataset(torch.utils.data.Dataset):
    """ torch-dataset used in training and prediction
    """
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
    """ text-collater used in training and prediction
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, list_items):
        # extract only tweets, tokenize them
        texts = [item[0] for item in list_items]
        batch = self.tokenizer(texts, truncation=True, padding=True)
        # extract labels (if we are training and not predicting)
        if 1 < len(list_items[0]):
            labels = [item[1] for item in list_items]
            batch["labels"] = labels
        batch = {key: torch.tensor(val) for key, val in batch.items()}
        return batch

def evaluation(model, dataloader, device):
    """ estimate accuracy of model

    Args:
        model : model to evaluate
        dataloader : data to evaluate on
        device : torch device

    Returns:
        accuracy of model on given data
    """
    model.eval()
    count_correct = 0.0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            preds = torch.argmax(outputs[1], dim=1)
            count_correct += torch.sum(preds == labels).item()
    accuracy = count_correct / len(dataloader.dataset)
    return accuracy