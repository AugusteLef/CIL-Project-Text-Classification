import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# import custom utils
import utils

def main(args):
    """ creates predictions for given data using given model

    Args:
        command-line arguments containing path to model, data etc.
    """
    # get data
    if args.verbose: print("reading data...")
    tweets = open(args.path_data)
    texts = tweets.readlines()

    # get tokenizers
    if args.verbose: print("loading tokenizers...")
    list_tokenizers = []
    for checkpoint in args.checkpoints_tokenizers:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
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
    for checkpoint in args.checkpoints_models:
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        list_models.append(model)
    model = utils.EnsembleModel(list_models)
    
    # use gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
   
    # inference
    preds = inference(model, dl, device)

    # write output
    if args.verbose: print("writing output...")
    preds = list(map(lambda x: -1 if x == 0 else 1, preds))
    df = pd.DataFrame(preds, index=list(range(1, len(preds)+1)))
    df.to_csv(args.path_output, header=["Prediction"], index_label="Id")

def inference(model, dl, device, args):
    if args.verbose: print("inference...")
    model.eval()
    results = []
    with torch.no_grad():
        for i, batch in enumerate(dl):
            inputs = batch["inputs"]
            inputs = utils.move_to_device(inputs, device)
            logits = model(**inputs)
            preds = torch.argmax(logits, dim=1)
            results += list(preds.cpu().numpy())
    return results

# creates prediction for given data using given model
if __name__ == "__main__":
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ["SCRATCH"], ".cache")

    parser = argparse.ArgumentParser(description='predict labels with given model')

    # command-line arguments
    # io
    parser.add_argument('path_data', type=str, 
        help='path to test data', action='store')
    parser.add_argument('path_output', type=str, 
        help='path to write output to', action='store')
    parser.add_argument('-ckptst', '--checkpoints_tokenizers', nargs="+", 
        help='path to pretrained tokenizers that should be used')
    parser.add_argument('-ckptsm', '--checkpoints_models', nargs="+", 
        help='path to pretrained models that should be used')
    parser.add_argument('-ckpt', '--checkpoint', type=str, 
        help='path to pretrained model that should be used')
    parser.add_argument('-v', '--verbose', dest='verbose', 
        help='want verbose output or not?', action='store_true')

    # inference
    parser.add_argument('-bs', '--batch_size', dest='batch_size', type=int, 
        help='size of batches for prediction', action='store', default=32)

    args = parser.parse_args()
    
    # predict labels
    main(args)

