import os
import torch
import argparse
import random
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

import utils
import models

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
    for config in args.configs:
        tokenizer = AutoTokenizer.from_pretrained(config)
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

    # use gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get model
    if args.verbose: print("loading models...")
    list_models = []
    for config, path_checkpoint in zip(args.configs, args.checkpoints):
        model_huggingface = AutoModelForSequenceClassification.from_pretrained(config, num_labels=2)
        model = models.HuggingfaceModel(model_huggingface)
        checkpoint = torch.load(path_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        list_models.append(model)
    model = models.EnsembleModel(list_models, args.freeze_models, size_hidden_state=2)
    model.to(device)

    # define loss function
    fn_loss = torch.nn.CrossEntropyLoss()

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
    parser.add_argument('-c', '--configs', nargs="+", 
        help='list of huggingface configs to use')
    parser.add_argument('-ckpts', '--checkpoints', nargs="+", 
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

