import os
import pandas as pd
import random
import torch
from transformers import AdamW
from nltk.corpus import wordnet
import random
from random import shuffle

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
			'ours', 'ourselves', 'you', 'your', 'yours',
			'yourself', 'yourselves', 'he', 'him', 'his',
			'himself', 'she', 'her', 'hers', 'herself',
			'it', 'its', 'itself', 'they', 'them', 'their',
			'theirs', 'themselves', 'what', 'which', 'who',
			'whom', 'this', 'that', 'these', 'those', 'am',
			'is', 'are', 'was', 'were', 'be', 'been', 'being',
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at',
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after',
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again',
			'further', 'then', 'once', 'here', 'there', 'when',
			'where', 'why', 'how', 'all', 'any', 'both', 'each',
			'few', 'more', 'most', 'other', 'some', 'such', 'no',
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
			'very', 's', 't', 'can', 'will', 'just', 'don',
			'should', 'now', '']


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

def get_synonyms(word):
    """
    Get synonyms of a word

    Args:
        word to get synonyms of

    Returns:
        list of synonyms of word
    """
    synonyms = set()

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)

    if word in synonyms:
        synonyms.remove(word)

    return list(synonyms)

def synonym_replacement(words, n):
    """
        replaces up to n synonyms of words in a string

        Args:
            word to get synonyms of

        Returns:
            list of synonyms of word
        """

    words = words.split()

    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0

    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)

        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1

        if num_replaced >= n:  # only replace up to n words
            break

    sentence = ' '.join(new_words)

    return sentence
    
def get_data_training(path_data_neg, path_data_pos, split):
    # get data
    if path_data_neg[-3:] == "txt":
        f_neg = open(path_data_neg)
        texts_neg = f_neg.readlines()
    else:
        df_neg = pd.read_csv(path_data_neg, keep_default_na=False)
        texts_neg = list(df_neg["tweet"])
    if path_data_pos[-3:] == "txt":
        f_pos = open(path_data_pos)
        texts_pos = f_pos.readlines()
    else:
        df_pos = pd.read_csv(path_data_pos, keep_default_na=False)
        texts_pos = list(df_pos["tweet"])

    # build train / test split
    random.shuffle(texts_neg) # should not be necessary but somehow is
    random.shuffle(texts_pos) # should not be necessary but somehow is
    split_neg = int(split*len(texts_neg))
    split_pos = int(split*len(texts_pos))
    texts_train = texts_neg[:split_neg] + texts_pos[:split_pos]
    labels_train = [0] * split_neg + [1] * split_pos
    texts_test = texts_neg[split_neg:] + texts_pos[split_pos:]
    labels_test = [0] * (len(texts_neg) - split_neg) + [1] * (len(texts_pos) - split_pos)

    return texts_train, labels_train, texts_test, labels_test

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
        # extract texts, tokenize them
        texts = [item[0] for item in list_items]
        inputs = self.tokenizer(texts, truncation=True, padding=True, max_length=512)
        inputs = {key: torch.tensor(val) for key, val in inputs.items()}
        batch = {"inputs": {"x": inputs}}

        # extract labels (if we are training and not predicting)
        if 1 < len(list_items[0]):
            labels = [item[1] for item in list_items]
            labels = torch.tensor(labels)
            batch["labels"] = labels
        
        return batch

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
            inputs = {key: torch.tensor(val) for key, val in inputs.items()}
            list_inputs.append(inputs)
        batch = {"inputs": {"x": list_inputs}}
        # extract labels (if we are training and not predicting)
        if 1 < len(list_items[0]):
            labels = [item[1] for item in list_items]
            labels = torch.tensor(labels)
            batch["labels"] = labels

        return batch

def move_to_device(x, device):
    """ move torch tensors in x to specified torch device

    Args:
        x: datastructure containing torch tensors
        device : torch device to move to

    Returns:
        same datastructure as x but with torch tensors moved to specified device
    """
    if torch.is_tensor(x):
        x = x.to(device)
    elif isinstance(x, dict):
        for key in x:
            x[key] = move_to_device(x[key], device)
    else:
        for idx in range(len(x)):
            x[idx] = move_to_device(x[idx], device)
    return x

def training(model, dataloader_train, dataloader_test, fn_loss, device, args):
    """ train the model and save checkpoints after every epoch

    Args:
        model : model to train
        dataloader_train : data to train on
        dataloader_test: data to evaluate on
        fn_loss: loss function to use
        device : torch device
        args: command line arguments

    Returns:
        None
    """
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)
    for epoch in range(args.epochs):
        model.train()
        avg_loss = 0.0
        for i, batch in enumerate(dataloader_train):
            inputs = batch["inputs"]
            labels = batch["labels"]
            inputs = move_to_device(inputs, device)
            labels = move_to_device(labels, device)
            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                preds = model(**inputs)
                loss = fn_loss(preds, labels)
                loss /= args.accumulation_size
            scaler.scale(loss).backward()
            avg_loss += loss.item()
            if (i + 1) % args.accumulation_size == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() 
                if args.verbose:
                    print(
                        "epoch %d/%d, batch %d/%d, avg. loss: %.3f" %
                        (
                            epoch+1,
                            args.epochs,
                            (i+1)//args.accumulation_size,
                            len(dataloader_train)//args.accumulation_size, avg_loss
                        )
                    )
                avg_loss = 0.0
        # evaluation
        accuracy = evaluation(model, dataloader_test, device)
        if args.verbose: print("evaluation...")
        print("accuracy: %.5f" % accuracy)
        # save model parameters to specified file
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "accuracy": accuracy,
        }
        path_checkpoint = os.path.join(args.dir_output, "checkpoint_%d" % (epoch+1))
        torch.save(checkpoint, path_checkpoint)

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
            inputs = batch["inputs"]
            labels = batch["labels"]
            inputs = move_to_device(inputs, device)
            labels = move_to_device(labels, device)
            logits = model(**inputs)
            preds = torch.argmax(logits, dim=1)
            count_correct += torch.sum(preds == labels).item()
    accuracy = count_correct / len(dataloader.dataset)
    return accuracy

def inference(model, dataloader, device):
    """ get models predictions for the given data

    Args:
        model : model to use
        dataloader : data to use
        device : torch device

    Returns:
        list of the models prediction for the given data
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs = batch["inputs"]
            inputs = move_to_device(inputs, device)
            logits = model(**inputs)
            preds_batch = torch.argmax(logits, dim=1)
            preds += list(preds_batch.cpu().numpy())
    return preds
