# inference script for ensemble 2 (using last transformer layer output)

import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# custom file imports
import utils_training_inference as utils
import models

def main(args):
    """ creates predictions for given data using given ensemble2 model

    Args:
        command-line arguments
    """
    # get data
    if args.verbose: print("reading data...")
    tweets = open(args.path_data)
    texts = tweets.readlines()

    # get tokenizers
    if args.verbose: print("loading tokenizers...")
    list_tokenizers = []
    for config in args.configs:
        tokenizer = AutoTokenizer.from_pretrained(config)
        list_tokenizers.append(tokenizer)
    
    # build dataloader
    collate_fn = utils.EnsembleCollator(list_tokenizers)
    ds = utils.TextDataset(texts) 
    dl = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    # build model
    if args.verbose: print("loading models...")
    list_models = []
    model_state_dict_bart = AutoModelForSequenceClassification.from_pretrained(args.configs[0]).state_dict()
    list_models.append(models.BartModelForEnsemble(model_state_dict_bart, list_tokenizers[0]))
    model_state_dict_bert = AutoModelForSequenceClassification.from_pretrained(args.configs[1]).state_dict()
    list_models.append(models.BertModelForEnsemble(model_state_dict_bert, list_tokenizers[1]))
    model_state_dict_bertweet = AutoModelForSequenceClassification.from_pretrained(args.configs[2]).state_dict()
    list_models.append(models.BertweetModelForEnsemble(model_state_dict_bertweet, list_tokenizers[2]))
    model_state_dict_xlnet = AutoModelForSequenceClassification.from_pretrained(args.configs[3]).state_dict()
    list_models.append(models.XLNetModelForEnsemble(model_state_dict_xlnet, list_tokenizers[3]))
    model = models.EnsembleModel(list_models, size_hidden_state=768)
    
    # use gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
   
    # inference
    if args.verbose: print("inference...")
    preds = utils.inference(model, dl, device)

    # create output directory
    dir_output = os.path.dirname(args.path_output)
    if not os.path.isdir(dir_output):
        os.makedirs(dir_output)

    # write output
    if args.verbose: print("writing output...")
    preds = list(map(lambda x: -1 if x == 0 else 1, preds))
    df = pd.DataFrame(preds, index=list(range(1, len(preds)+1)))
    df.to_csv(args.path_output, header=["Prediction"], index_label="Id")

# creates prediction for given data using given model
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='predict labels with given model')

    # command-line arguments
    # io
    parser.add_argument('path_data', type=str, 
        help='path to test data', action='store')
    parser.add_argument('path_output', type=str, 
        help='path to write output to', action='store')
    parser.add_argument('-v', '--verbose', dest='verbose', 
        help='want verbose output or not?', action='store_true')
    parser.add_argument('-c', '--configs', nargs="+", 
        help='list of huggingface configs to use')
    parser.add_argument('-ckpt', '--checkpoint', type=str, 
        help='path to pretrained model that should be used')

    # inference parameters
    parser.add_argument('-bs', '--batch_size', dest='batch_size', type=int, 
        help='size of batches for prediction', action='store', default=32)

    args = parser.parse_args()
    
    # predict labels
    main(args)

