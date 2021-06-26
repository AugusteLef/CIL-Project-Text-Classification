import os
import torch
import argparse
import random
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

import utils

class EnsembleTokenizer():
    def __init__(self, list_tokenizers):
        self.list_tokenizers = list_tokenizers

    def __call__(self, batch, truncation=True, padding=True):
        apply_tokenizer = lambda t: t(batch, truncation=truncation, padding=padding)
        list_batches = list(map(apply_tokenizer, self.list_tokenizers))
        return list_batches

class EnsembleModel(torch.nn.Module):
    def __init__(self, list_models):
        super(EnsembleModel, self).__init__()
        self.list_models = list_models
        self.layer_linear = torch.nn.Linear(
            in_features=2*len(list_models),
            out_features=2,
        )
        self.layer_softmax = torch.nn.Softmax()
    
    def forward(self, x):
        list_logits = list(map(lambda m: m(x), self.list_models))
        tmp = torch.cat(list_logits)
        logits = self.layer_linear(tmp)
        return self.softmax(logits)

class EnsembleCollator():
    """ text-collater used in training and prediction
    """
    def __init__(self, list_tokenizers):
        self.list_tokenizers = list_tokenizers

    def __call__(self, list_items):
        # extract only tweets, tokenize them
        texts = [item[0] for item in list_items]
        list_batches = []
        for tokenizer in self.list_tokenizers:
            batch = tokenizer(texts, truncation=True, padding=True)
            list_batches.append(batch)
        print(batch)
        # extract labels (if we are training and not predicting)
        if 1 < len(list_items[0]):
            labels = [item[1] for item in list_items]
            for batch in list_batches:
                batch["labels"] = labels
        for i in range(len(list_batches)):
            list_batches[i] = {key: torch.tensor(val) for key, val in list_batches[i].items()}
        return list_batches
        
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
    print(next(iter(dl_train)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.verbose: print("loading models...")
    list_models = []
    for checkpoint in args.checkpoints_models:
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
        list_models.append(model)
    model = EnsembleModel(list_models)
    model.to(device)
   
if __name__ == "__main__":
    #os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ["SCRATCH"], ".cache")

    parser = argparse.ArgumentParser(description='train ensemble model on data')
    
    # command line arguments
    parser.add_argument('neg_data', type=str, 
        help='path to negative training data', action='store')
    parser.add_argument('pos_data', type=str, 
        help='path to positive training data', action='store')
    parser.add_argument('dir_output', type=str, 
        help='path where model should be store', action='store')
    parser.add_argument('-ckptst', '--checkpoints_tokenizers', nargs="+", 
        help='path to pretrained tokenizers that should be used')
    parser.add_argument('-ckptsm', '--checkpoints_models', nargs="+", 
        help='path to pretrained models that should be used')
    parser.add_argument('-v', '--verbose', 
        help='want verbose output or not?', action='store_true')
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

