# training script for ensemble 2 (using last transformer layer output)

import os
import torch
import argparse
from transformers import AutoTokenizer

# custom file imports 
import utils_training_inference as utils
import models

# this script only works with this config-list!
list_configs = [
    "facebook/bart-base",
    "bert-base-uncased",
    "vinai/bertweet-base",
    "xlnet-base-cased",
]

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

    # get tokenizers
    if args.verbose: print("loading tokenizers...")
    list_tokenizers = []
    for config in list_configs:
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
    checkpoint_bart = torch.load(args.checkpoints[0], map_location=device)
    list_models.append(models.BartModelForEnsemble(checkpoint_bart["model_state_dict"], list_tokenizers[0]))
    checkpoint_bert = torch.load(args.checkpoints[1], map_location=device)
    list_models.append(models.BertModelForEnsemble(checkpoint_bert["model_state_dict"], list_tokenizers[1]))
    checkpoint_bertweet = torch.load(args.checkpoints[2], map_location=device)
    list_models.append(models.BertweetModelForEnsemble(checkpoint_bertweet["model_state_dict"], list_tokenizers[2]))
    checkpoint_xlnet = torch.load(args.checkpoints[3], map_location=device)
    list_models.append(models.XLNetModelForEnsemble(checkpoint_xlnet["model_state_dict"], list_tokenizers[3]))
    model = models.EnsembleModel(list_models, args.freeze_models, 768, args.dense_layers)
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
    parser.add_argument('-ckpts', '--checkpoints', nargs="+", 
        help='path to pretrained models that should be used; order must be bart, bert, bertweet, xlnet')
    parser.add_argument('-fm', '--freeze_models', 
        help='set to freeze submodules', action='store_true')

    # model arguments
    parser.add_argument('-dl', '-dense_layers', type=int, choices=range(1,3), 
        help='number of dense layers to use in ensemble', default=1)

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

