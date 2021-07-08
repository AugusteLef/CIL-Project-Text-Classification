import os
import torch
import argparse
import random
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

import utils

class EnsembleModel(torch.nn.Module):
    def __init__(self, list_models, freeze_models=False):
        super(EnsembleModel, self).__init__()
        self.list_models = torch.nn.ModuleList(list_models)
        self.layer_linear = torch.nn.Linear(
            in_features=2*len(list_models),
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
    
def main(args):
    # get data
    if args.verbose: print("reading data...")
    texts_train, labels_train, texts_test, labels_test = utils.get_data_training(args.neg_data, args.pos_data, args.split)
    if args.verbose:
        print("%d training samples" % len(texts_train))
        print("%d test samples" % len(texts_test))

    # get tokenizers
    if args.verbose: print("loading tokenizers...")
    list_tokenizers = []
    for checkpoint in args.checkpoints_tokenizers:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        list_tokenizers.append(tokenizer)
    
    # build dataloaders
    collate_fn = utils.EnsembleCollator(list_tokenizers)
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

    # get model
    if args.verbose: print("loading models...")
    list_models = []
    for checkpoint in args.checkpoints_models:
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
        list_models.append(model)
    model = EnsembleModel(list_models, args.freeze_models)

    # define loss function
    fn_loss = torch.nn.CrossEntropyLoss()

    # use gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create output directory
    if not os.path.isdir(args.dir_output):
        os.makedirs(args.dir_output)

    # train
    if args.verbose: print("training...")
    utils.training(model, dl_train, dl_test, fn_loss, device, args)
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train ensemble model on data')
    
    # command line arguments
    # io arguments
    parser.add_argument('neg_data', type=str, 
        help='path to negative training data', action='store')
    parser.add_argument('pos_data', type=str, 
        help='path to positive training data', action='store')
    parser.add_argument('dir_output', type=str, 
        help='directory where model checkpoints should be stored', action='store')
    parser.add_argument('-v', '--verbose', 
        help='set for verbose output', action='store_true')
    parser.add_argument('-ckptst', '--checkpoints_tokenizers', nargs="+", 
        help='path to pretrained tokenizers that should be used')
    parser.add_argument('-ckptsm', '--checkpoints_models', nargs="+", 
        help='path to pretrained models that should be used')
    parser.add_argument('-fm', '--freeze_models', 
        help='set to freeze submodules', action='store_true')

    # training arguments
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

    # parse
    args = parser.parse_args()

    # set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # start training
    main(args)

