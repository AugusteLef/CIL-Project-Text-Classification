import os
import torch
import argparse
import random
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from pathlib import Path

class EncodingsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# used for testing on imdb dataset
def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir == "neg" else 1)
    return texts, labels

def main(args):
    # get the data
    if args.verbose: print("reading data...")
    f_neg = open(args.neg_data)
    texts_neg = f_neg.readlines()
    f_pos = open(args.pos_data)
    texts_pos = f_pos.readlines()

    # create train / test split
    n_neg = int(args.split*len(texts_neg))
    n_pos = int(args.split*len(texts_pos))
    random.shuffle(texts_neg)
    random.shuffle(texts_pos)
    texts_train = texts_neg[:n_neg] + texts_pos[:n_pos]
    labels_train = [0] * n_neg + [1] * n_pos
    texts_test = texts_neg[n_neg:] + texts_pos[n_pos:]
    labels_test = [0] * (len(texts_neg) - n_neg) + [1] * (len(texts_pos) - n_pos)
    
    # was used for testing on imdb
    #texts_train, labels_train = read_imdb_split(os.path.join(DIR_DATA, 'aclImdb/train'))
    #texts_test, labels_test = read_imdb_split(os.path.join(DIR_DATA, 'aclImdb/test'))

    # get the tokenizer
    if args.verbose: print("loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model) 

    # apply tokenizer
    if args.verbose: print("tokenizing data...")
    encodings_train = tokenizer(texts_train, truncation=True, padding=True)
    encodings_test = tokenizer(texts_test, truncation=True, padding=True)
    
    # build dataloader
    ds_train = EncodingsDataset(encodings_train, labels_train) 
    ds_test = EncodingsDataset(encodings_test, labels_test) 
    dl_train = torch.utils.data.DataLoader(
        dataset=ds_train,
        batch_size=int(args.batch_size/args.accumulation_size),
        shuffle=True,
        num_workers=4
    )
    dl_test = torch.utils.data.DataLoader(
        dataset=ds_test,
        batch_size=args.batch_size
    )
    
    # use gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load pretrained model
    model = BertForSequenceClassification.from_pretrained(args.pretrained_model)
    model = torch.nn.DataParallel(model)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # train
    if args.verbose: print("training...")
    model.train()
    for epoch in range(args.epochs):
        avg_loss = 0.0
        for i, batch in enumerate(dl_train):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss /= args.accumulation_size
            loss.backward()
            avg_loss += loss.item()
            if (i + 1) % args.accumulation_size == 0:
                optimizer.step()
                optimizer.zero_grad() 
                if args.verbose:
                    print(
                        "epoch %d/%d, batch %d/%d, avg. loss: %.3f" %
                        (epoch+1, args.epochs, i, len(dl_train), avg_loss)
                    )
                avg_loss = 0.0
    
    # evaluate i.e. generate accuracy estimation
    if args.verbose: print("evaluation...")
    model.eval()
    count_correct = 0.0
    with torch.no_grad():
        for i, batch in enumerate(dl_test):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            preds = torch.argmax(outputs[1], dim=1)
            count_correct += torch.sum(preds == labels).item()
    accuracy = count_correct / len(ds_test)
    print("accuracy: %.3f" % accuracy)

    # save model parameters to specified file
    dir_model = os.path.dirname(args.model_destination)
    if dir_model != "" and not os.path.exists(dir_model):
        os.makedirs(dir_model)
    torch.save(model.state_dict(), args.model_destination)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train pretrained BERT model on data')

    parser.add_argument('neg_data', type=str, 
        help='path to negative training data', action='store')
    parser.add_argument('pos_data', type=str, 
        help='path to positive training data', action='store')
    parser.add_argument('-pm', '--pretrained_model', dest='pretrained_model', type=str, 
        help='path to pretrained model that should be used', default='Pretrained/bert-base-uncased')
    parser.add_argument('model_destination', type=str, 
        help='path where model should be store', action='store')
    parser.add_argument('-v', '--verbose', dest='verbose', 
        help='want verbose output or not?', action='store_true')
    parser.add_argument('-e', '-epochs', dest='epochs', type=int, 
        help='number of epochs to train', action='store', default=3)
    parser.add_argument('-bs', '--batch_size', dest='batch_size', type=int, 
        help='size of batches for training', action='store', default=16)
    parser.add_argument('-as', '--accumulation_size', dest='accumulation_size', type=int, 
        help='reduces memory usage, if larger', action='store', default=16)
    parser.add_argument('--seed', dest='seed', type=int, 
        help='fix random seeds', action='store', default=42)
    parser.add_argument('--split', dest='split', type=float, 
        help='define train/test split, number between 0 and 1', action='store', default=0.8)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    main(args)

