import os
import torch
import argparse
import random
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

import utils

class EnsembleModel(torch.nn.Module):
    def __init__(self, list_models):
        super(EnsembleModel, self).__init__()
        self.list_models = torch.nn.ModuleList(list_models)
        self.layer_linear = torch.nn.Linear(
            in_features=2*len(list_models),
            out_features=2,
        )
    
    def forward(self, x):
        list_logits = []
        for i in range(len(self.list_models)):
            model = self.list_models[i]
            logits = model(**x[i])[0] # TODO: wrapper for models
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
            inputs = tokenizer(texts, truncation=True, padding=True)
            inputs = {key: torch.tensor(val) for key, val in inputs.items()} # TODO: wrapper for tokenizers
            list_inputs.append(inputs)
        batch = {"inputs": list_inputs}

        # extract labels (if we are training and not predicting)
        if 1 < len(list_items[0]):
            labels = [item[1] for item in list_items]
            labels = torch.tensor(labels)
            batch["labels"] = labels

        return batch
        
def main(args):
    if args.verbose: print("reading data...")
    if args.neg_data[-3:] == "txt":
        f_neg = open(args.neg_data)
        texts_neg = f_neg.readlines()
    else:
        df_neg = pd.read_csv(args.neg_data, keep_default_na=False)
        texts_neg = list(df_neg["tweet"])
    if args.pos_data[-3:] == "txt":
        f_pos = open(args.pos_data)
        texts_pos = f_pos.readlines()
    else:
        df_pos = pd.read_csv(args.pos_data, keep_default_na=False)
        texts_pos = list(df_pos["tweet"])

    # create train / test split
    n_neg = int(args.split*len(texts_neg))
    n_pos = int(args.split*len(texts_pos))
    random.shuffle(texts_neg)
    random.shuffle(texts_pos)
    texts_train = texts_neg[:n_neg] + texts_pos[:n_pos]
    labels_train = [0] * n_neg + [1] * n_pos
    texts_test = texts_neg[n_neg:] + texts_pos[n_pos:]
    labels_test = [0] * (len(texts_neg) - n_neg) + [1] * (len(texts_pos) - n_pos)
    if args.verbose:
        print("%d training samples" % (n_neg + n_pos))
        print("%d test samples" % (len(texts_neg) - n_neg + len(texts_pos) - n_pos))

    if args.verbose: print("loading tokenizers...")
    list_tokenizers = []
    for checkpoint in args.checkpoints_tokenizers:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        list_tokenizers.append(tokenizer)
    collate_fn = EnsembleCollator(list_tokenizers)

    ds_train = utils.TextDataset(texts_train, labels_train) 
    ds_test = utils.TextDataset(texts_test, labels_test) 
    dl_train = torch.utils.data.DataLoader(
        dataset=ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    dl_test = torch.utils.data.DataLoader(
        dataset=ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    if args.verbose: print("loading models...")
    list_models = []
    for checkpoint in args.checkpoints_models:
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
        list_models.append(model)
    model = EnsembleModel(list_models)

    fn_loss = torch.nn.CrossEntropyLoss()

    train(model, dl_train, dl_test, fn_loss, args)

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
            inputs = batch["inputs"]
            labels = batch["labels"]
            inputs = move_to_device(inputs, device)
            labels = move_to_device(labels, device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            count_correct += torch.sum(preds == labels).item()
    accuracy = count_correct / len(dataloader.dataset)
    return accuracy

def train(model, dl_train, dl_test, fn_loss, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    if args.verbose: print("training...")
    for epoch in range(args.epochs):
        model.train()
        avg_loss = 0.0
        for i, batch in enumerate(dl_train):
            inputs = batch["inputs"]
            labels = batch["labels"]
            inputs = move_to_device(inputs, device)
            labels = move_to_device(labels, device)
            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                preds = model(inputs)
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
                        (epoch+1, args.epochs, i//args.accumulation_size, len(dl_train)//args.accumulation_size, avg_loss)
                    )
                avg_loss = 0.0
        # evaluation
        if args.verbose: print("evaluation...")
        print("accuracy: %.5f" % evaluation(model, dl_test, device))
        # save model parameters to specified file
        model.save_pretrained(os.path.join(args.dir_output, "checkpoint_%d" % (epoch+1)))
   
if __name__ == "__main__":
    #os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ["SCRATCH"], ".cache")

    parser = argparse.ArgumentParser(description='train ensemble model on data')
    
    # command line arguments
    # io
    parser.add_argument('neg_data', type=str, 
        help='path to negative training data', action='store')
    parser.add_argument('pos_data', type=str, 
        help='path to positive training data', action='store')
    parser.add_argument('dir_output', type=str, 
        help='directory where model checkpoints should be stored', action='store')
    parser.add_argument('-ckptst', '--checkpoints_tokenizers', nargs="+", 
        help='path to pretrained tokenizers that should be used')
    parser.add_argument('-ckptsm', '--checkpoints_models', nargs="+", 
        help='path to pretrained models that should be used')
    parser.add_argument('-v', '--verbose', 
        help='want verbose output or not?', action='store_true')

    # training
    parser.add_argument('-e', '-epochs', dest='epochs', type=int, 
        help='number of epochs to train', action='store', default=3)
    parser.add_argument('-bs', '--batch_size', dest='batch_size', type=int, 
        help='size of batches for training', action='store', default=8)
    parser.add_argument('-as', '--accumulation_size', dest='accumulation_size', type=int, 
        help='reduces memory usage, if larger', action='store', default=4)
    parser.add_argument('--seed', dest='seed', type=int, 
        help='fix random seeds', action='store', default=42)
    parser.add_argument('--split', dest='split', type=float, 
        help='define train/test split, number between 0 and 1', action='store', default=0.8)
    parser.add_argument('-mp', '--mixed_precision', dest='mixed_precision',
        help='set to enable mixed precision training', action='store_true', default=False)

    args = parser.parse_args()

    # set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # start training
    main(args)

