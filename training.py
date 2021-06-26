import os
import torch
import argparse
import random
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

# import custom utils
import utils

def main(args):
    """ main training routine

    Args:
        args: command line arguments containing paths to training data, pretrained model, output location etc.
    """
    # get the data (can also handle csv with explicit labels)
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

    # XLNET need sep and cls tags at the end of each tweet
    if args.XLNET:
        texts_pos = utils.XLNET_tweets_transformation(texts_pos)
        texts_neg = utils.XLNET_tweets_transformation(texts_neg)

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
    
    # get the tokenizer
    if args.verbose: print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model) 
    collate_fn = utils.TextCollator(tokenizer, xlnet=args.XLNET)
    
    # build dataloader
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

    # load pretrained model
    model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=2) # what was the argument num_labels=2 ?
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    # train
    if args.verbose: print("training...")
    for epoch in range(args.epochs):
        model.train()
        avg_loss = 0.0
        for i, batch in enumerate(dl_train):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
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
        print("accuracy: %.5f" % utils.evaluation(model, dl_test, device))
        # save model parameters to specified file
        model.save_pretrained(os.path.join(args.model_destination, "checkpoint_%d" % (epoch+1)))

# this script executes a full training routine according to command-line arguments
if __name__ == "__main__":
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ["SCRATCH"], ".cache")

    parser = argparse.ArgumentParser(description='train pretrained model on data')
    
    # command line arguments
    parser.add_argument('neg_data', type=str, 
        help='path to negative training data', action='store')
    parser.add_argument('pos_data', type=str, 
        help='path to positive training data', action='store')
    parser.add_argument('model_destination', type=str, 
        help='path where model should be store', action='store')
    parser.add_argument('-pm', '--pretrained_model', dest='pretrained_model', type=str, 
        help='path to pretrained model that should be used', default='Pretrained_Models/bart-base')
    parser.add_argument('-v', '--verbose', dest='verbose', 
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
    parser.add_argument('-x', '--XLNET', dest='XLNET', 
        help='must set this flag when using XLNET', action='store_true')
    parser.add_argument('-mp', '--mixed_precision', dest='mixed_precision',
        help='set to enable mixed precision training', action='store_true', default=False)
    args = parser.parse_args()

    # set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # start training
    main(args)

