# utitily functions used in training and inference loops

# imports
import os
import random
import torch
import pandas as pd
import numpy as np
from transformers import AdamW

def seed_everything(seed=1, pytorch=True):
    """ Seed everything.
    """   
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    if pytorch:
        torch.manual_seed(seed)
    
def get_data_training(path_data_neg, path_data_pos, split):
    """ load positive and negative data and create training/validation split
    """
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
    random.shuffle(texts_neg) 
    random.shuffle(texts_pos) 
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
        inputs = self.tokenizer(texts, truncation=True, padding=True, max_length=120)
        inputs = {key: torch.tensor(val) for key, val in inputs.items()}
        batch = {"inputs": {"x": inputs}}

        # extract labels (if we are training and not predicting)
        if 1 < len(list_items[0]):
            labels = [item[1] for item in list_items]
            labels = torch.tensor(labels)
            batch["labels"] = labels
        
        return batch

class EnsembleCollator():
    """ ensemble-collater used in training and prediction
    """
    def __init__(self, list_tokenizers):
        self.list_tokenizers = list_tokenizers

    def __call__(self, list_items):
        # extract only tweets, tokenize them
        texts = [item[0] for item in list_items]
        list_inputs = []
        for tokenizer in self.list_tokenizers:
            inputs = tokenizer(texts, truncation=True, padding=True, max_length=120)
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
    optimizer = AdamW(model.parameters(), lr=2e-5)
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
