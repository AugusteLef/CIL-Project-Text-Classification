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
    
def XLNET_tweets_transformation(tweets):
    """ XLNET needs sep and cls tags at the end of each tweet

    Args:
        list of tweets

    Returns: 
        list of processed tweets
    """
    out = []
    for tweet in tweets:
        tweet = tweet + '[SEP] [CLS]'
        out.append(tweet)
    return out

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
    def __init__(self, tokenizer, xlnet=False):
        self.tokenizer = tokenizer
        self.xlnet = xlnet

    def __call__(self, list_items):
        # extract only tweets, tokenize them
        texts = [item[0] for item in list_items]
        batch = 0 # TODO: is this variable declaration needed?
        if self.xlnet:
            batch = self.tokenizer(texts, truncation=True, padding=True, max_length=140)
        else:
            batch = self.tokenizer(texts, truncation=True, padding=True)
        # extract labels (if we are training and not predicting)
        if 1 < len(list_items[0]):
            labels = [item[1] for item in list_items]
            batch["labels"] = labels
        batch = {key: torch.tensor(val) for key, val in batch.items()}
        return batch

class EnsembleModel(torch.nn.Module):
    def __init__(self, list_models, freeze_models=False):
        super(EnsembleModel, self).__init__()
        self.list_models = torch.nn.ModuleList(list_models)
        self.layer_linear = torch.nn.Linear(
            in_features=9, # TODO: change back to 2*len(list_models) 
            out_features=2,
        )
        if freeze_models:
            for model in self.list_models:
                for param in model.parameters():
                    param.requires_grad = False
    
    def forward(self, x):
        list_logits = []
        for i in range(len(self.list_models)):
            model = self.list_models[i]
            logits = model(**x[i])[0] # TODO: wrapper for models because of [0]
            list_logits.append(logits)
        tmp = torch.cat(list_logits, axis=1)
        logits = self.layer_linear(tmp)
        return logits
    
class EnsembleCollator():
    """ text-collater used in training and prediction
    """
    def __init__(self, list_tokenizers):
        self.list_tokenizers = list_tokenizers

    def __call__(self, list_items):
        # extract only tweets, tokenize them
        texts = [item[0] for item in list_items]
        list_inputs = []
        for tokenizer in self.list_tokenizers:
            inputs = tokenizer(texts, truncation=True, padding=True, max_length=512)
            inputs = {key: torch.tensor(val) for key, val in inputs.items()} # TODO: wrapper for tokenizers
            list_inputs.append(inputs)
        batch = {"inputs": {"x": list_inputs}}

        # extract labels (if we are training and not predicting)
        if 1 < len(list_items[0]):
            labels = [item[1] for item in list_items]
            labels = torch.tensor(labels)
            batch["labels"] = labels

        return batch

def move_to_device(x, device):
    if torch.is_tensor(x):
        x = x.to(device)
    elif isinstance(x, dict):
        for key in x:
            x[key] = move_to_device(x[key], device)
    else:
        for idx in range(len(x)):
            x[idx] = move_to_device(x[idx], device)
    return x

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
