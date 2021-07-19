# training script for huggingface models

import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# custom file imports 
import utils_training_inference as utils
import models

def main(args):
    """ main training routine

    Args:
        args: command line arguments
    """
    # get data
    if args.verbose: print("reading data...")
    texts_train, labels_train, texts_test, labels_test = utils.get_data_training(args.neg_data, args.pos_data, args.split)
    if args.verbose:
        print("%d training samples" % len(texts_train))
        print("%d test samples" % len(texts_test))
    
    # get tokenizer
    if args.verbose: print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.config) 
    
    # build dataloaders
    collate_fn = utils.TextCollator(tokenizer)
    ds_train = utils.TextDataset(texts_train, labels_train) 
    ds_test = utils.TextDataset(texts_test, labels_test) 
    dl_train = torch.utils.data.DataLoader(
        dataset=ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    dl_test = torch.utils.data.DataLoader(
        dataset=ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    # get model
    if args.verbose: print("loading model...")
    model_huggingface = AutoModelForSequenceClassification.from_pretrained(args.config, num_labels=2)
    model = models.HuggingfaceModel(model_huggingface)
    model.resize_token_embeddings(len(tokenizer))
    
    # define loss function
    fn_loss = torch.nn.CrossEntropyLoss()

    # use gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # create output directory
    if not os.path.isdir(args.dir_output):
        os.makedirs(args.dir_output)

    # train
    if args.verbose: print("training...")
    utils.training(model, dl_train, dl_test, fn_loss, device, args)

# this script executes a full training routine according to command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train pretrained model on data')
    
    # command line arguments
    # io arguments
    parser.add_argument('neg_data', type=str, 
        help='path to negative training data', action='store')
    parser.add_argument('pos_data', type=str, 
        help='path to positive training data', action='store')
    parser.add_argument('dir_output', type=str, 
        help='directory where checkpoints should be stored', action='store')
    parser.add_argument('-v', '--verbose', dest='verbose', 
        help='want verbose output or not?', action='store_true')
    parser.add_argument('-c', '--config', type=str, 
        help='huggingface config to use')

    # training arguments
    parser.add_argument('-e', '-epochs', dest='epochs', type=int, 
        help='number of epochs to train', action='store', default=3)
    parser.add_argument('-bs', '--batch_size', dest='batch_size', type=int, 
        help='size of batches for training', action='store', default=8)
    parser.add_argument('-as', '--accumulation_size', dest='accumulation_size', type=int, 
        help='reduces memory usage, if larger', action='store', default=4)
    parser.add_argument('--seed', dest='seed', type=int, 
        help='fix random seeds', action='store', default=1)
    parser.add_argument('--split', dest='split', type=float, 
        help='define train/test split, number between 0 and 1', action='store', default=0.8)
    parser.add_argument('-mp', '--mixed_precision', dest='mixed_precision',
        help='set to enable mixed precision training', action='store_true', default=False)

    # parse 
    args = parser.parse_args()

    # set seeds
    utils.seed_everything(seed = args.seed)

    # start training
    main(args)

