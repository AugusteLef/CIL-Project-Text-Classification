import os
import torch
import argparse
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification
import pandas as pd
import utils

def main(args):
    # get the data
    if args.verbose: print("reading data...")
    df = pd.read_csv(args.path_data, keep_default_na=False)
    texts = list(df["tweet"])
    
    # get the tokenizer
    if args.verbose: print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.dir_tokenizer) 
    collate_fn = utils.TextCollator(tokenizer)

    # build dataloader
    ds = utils.TextDataset(texts) 
    dl = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
        )

    # use gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load pretrained model
    model = BertForSequenceClassification.from_pretrained(args.dir_model)
    model.to(device)
    
    # inference
    if args.verbose: print("inference...")
    model.eval()
    results = []
    with torch.no_grad():
        for i, batch in enumerate(dl):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs[0], dim=1)
            tmp = list(preds.cpu().numpy())
            tmp = list(map(lambda x: -1 if x == 0 else 1, tmp))
            results += tmp
    
    # writing output
    if args.verbose: print("writing output...")
    df = pd.DataFrame(results, index=list(range(1, len(results)+1)))
    df.to_csv(args.path_output, header=["Prediction"], index_label="Id")

if __name__ == "__main__":
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ["SCRATCH"], ".cache")

    parser = argparse.ArgumentParser(description='train pretrained BERT model on data')

    parser.add_argument('path_data', type=str, 
        help='path to test data', action='store')
    parser.add_argument('path_output', type=str, 
        help='path to write output to', action='store')
    parser.add_argument('-v', '--verbose', dest='verbose', 
        help='want verbose output or not?', action='store_true')
    parser.add_argument('-dt', '--dir_tokenizer', dest='dir_tokenizer', type=str, 
        help='directory containing pretrained tokenizer')
    parser.add_argument('-dm', '--dir_model', dest='dir_model', type=str, 
        help='directory containing pretrained model')
    parser.add_argument('-bs', '--batch_size', dest='batch_size', type=int, 
        help='size of batches for training', action='store', default=16)
    parser.add_argument('--seed', dest='seed', type=int, 
        help='fix random seeds', action='store', default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    main(args)

