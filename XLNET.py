#TO TRAIN THE XLNET MODEL ON FULL DATASET & SAVE THE MODEL

import pandas as pd
import numpy as np
import transformers
#import preprocessing as pp
#from pytorch_transformers import XLNetTokenizer,XLNetForSequenceClassification
from transformers import XLNetTokenizer,XLNetForSequenceClassification
from sklearn.model_selection import train_test_split
from pytorch_transformers import AdamW
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
import torch
import re
import torch.nn as nn
from torch.nn import DataParallel
from torch.utils.data import TensorDataset,DataLoader,RandomSampler,SequentialSampler

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


#global variables
POSITIVE_PATH = args.pos_data
NEGATIVE_PATH  = args.neg_data
TEST_PATH = args.test_data
OUTPUT_PREDS_PATH = 'XLNET_test_preds_Stop.txt'

PRETRAINED_PATH = args.pretrained_model
N_EPOCHS = args.epochs # 3 by default
BATCH_SIZE = args.batch_size # 16 by default
TEST_SIZE = 1 - args.split # 0.8 by default

def load_dataframe(path: str) -> pd.DataFrame:
    """Load a dataframe containing tweet and labels

    Args:
        path (str): path to the dataframe (.txt)

    Returns:
        DataFrame: the loaded dataframe
    """
    df = pd.read_csv(path, sep = ',')
    return df

def concat_data(dset1, dset2):
    df = pd.concat([dset1, dset2])
    print('Concatenated data')
    return df

#XLNET need sep and cls tags at the end of each tweet
def XLNET_tweets_transformation(df):
    tweets = []
    for tweet in df['tweet']:
        tweet = str(tweet)+'[SEP] [CLS]'
        tweets.append(tweet)
    return tweets

def get_max_length_ids(ids):
    maxlen = len(ids[0])
    for i in ids:
        if (len(i)>maxlen):
            maxlen = len(i)
    return maxlen

def accuracy(preds, labels):
    #1 is positive 0 is negative
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(labels)):
        if(preds[i]==labels[i]):
            if preds[i] == 1:
                TP += 1
            else:
                TN += 1
        else:
            if preds[i] == 1:
                FP += 1
            else:
                FN += 1
    return ((TP+TN)/(TP+TN+FP+FN))*100


def main():
    #load datas
    pos_df = load_dataframe(POSITIVE_PATH)
    neg_df = load_dataframe(NEGATIVE_PATH)
    pos_df['label'] = pos_df['label'].apply(lambda x: 1)
    neg_df['label'] = neg_df['label'].apply(lambda x: 0)

    test_df = load_dataframe(TEST_PATH)

    #concat datas
    all_tweet_df = concat_data(pos_df, neg_df)

    #XLNET tranformation
    tweets = XLNET_tweets_transformation(all_tweet_df)
    test_tweets = XLNET_tweets_transformation(test_df)

    #Tokenization
    tokenizer = XLNetTokenizer.from_pretrained(PRETRAINED_PATH,do_lower_case=True)

    tokenized_tweets = [tokenizer.tokenize(tweet) for tweet in tweets]
    tokenized_test_tweets = [tokenizer.tokenize(test_tweet) for test_tweet in test_tweets]

    #ids and label
    ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_tweets]
    labels = all_tweet_df['label'].values
    test_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_test_tweets]

    #get max length and pad ids
    MAX_LENGTH = get_max_length_ids(ids)
    padded_ids = pad_sequences(ids, maxlen=MAX_LENGTH, dtype="long", truncating="post", padding="post")

    MAX_LENGTH_TEST = get_max_length_ids(test_ids)
    padded_test_ids = pad_sequences(test_ids, maxlen=MAX_LENGTH_TEST, dtype="long", truncating="post", padding="post")


    # build dataloader
    #ds_train = EncodingsDataset(encodings_train, labels_train) 
    #ds_test = EncodingsDataset(encodings_test, labels_test) 
    #loader = torch.utils.data.DataLoader(
    #    dataset=ds_train,
    #    batch_size=args.batch_size//args.accumulation_size,
    #    shuffle=True,
    #    num_workers=4
    #)
    #test_loader = torch.utils.data.DataLoader(
    #    dataset=ds_test,
    #    batch_size=args.batch_size
    #)

    #splitdataset 
    xtrain,xtest,ytrain,ytest = train_test_split(padded_ids, labels, test_size=0)

    transform dataset to torch.tensor format
    Xtrain = torch.tensor(xtrain)
    Ytrain = torch.tensor(ytrain)
    Xtest = torch.tensor(xtest)
    Ytest = torch.tensor(ytest)

    train_data = TensorDataset(Xtrain,Ytrain)
    test_data = TensorDataset(Xtest,Ytest)
    loader = DataLoader(train_data,batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data,batch_size=BATCH_SIZE)

    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #set the model
    model = XLNetForSequenceClassification.from_pretrained(PRETRAINED_PATH)

    #parallel
    model = nn.DataParallel(model)

    #model to device
    model.to(device)

    #need nvidia driver ?
    model.cuda()

    #set optimizer and loss function
    optimizer = AdamW(model.parameters(),lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    #Training
    no_train = 0
    for epoch in range(N_EPOCH):
        print('Epoch: ' + str(epoch))
        model.train()
        loss1 = []
        steps = 0
        train_loss = []
        l = []
        for inputs,labels1 in loader :
            inputs.to(device)
            labels1.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs[0],labels1.to(device)).to(device)
            #logits = outputs[1]
            #ll=outp(loss)
            [train_loss.append(p.item()) for p in torch.argmax(outputs[0],axis=1).flatten() ]#our predicted 
            [l.append(z.item()) for z in labels1]# real labels
            loss.backward()
            optimizer.step()
            loss1.append(loss.item())
            no_train += inputs.size(0)
            steps += 1
            print('Steps : ' + str(steps))
        print("Current Loss is : {} Step is : {} number of Example : {} Accuracy : {}".format(loss.item(),epoch,no_train,accuracy(train_loss,l)))
        model.module.save_pretrained(os.path.join(args.model_destination, "checkpoint_%d" % epoch))


    # #testing
    # model.eval()
    # acc = []
    # lab = []
    # t = 0
    # for inp,lab1 in test_loader:
    #     inp.to(device)
    #     lab1.to(device)
    #     t+=lab1.size(0)
    #     outp1 = model(inp.to(device))
    #     [acc.append(p1.item()) for p1 in torch.argmax(outp1[0],axis=1).flatten() ]
    #     [lab.append(z1.item()) for z1 in lab1]
    # print("Total Examples : {} Accuracy {}".format(t,accuracy(acc,lab)))

    # to_save = []
    # to_save.append(acc)
    # to_save.append(lab)
    # to_save.append([accuracy(acc,lab)])
    # np.savetxt(OUTPUT_PREDS_PATH, np.asarray(to_save), delimiter=',', fmt='%s')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train pretrained XLNET model on data')

    parser.add_argument('neg_data', type=str, 
        help='path to negative training data', action='store')
    parser.add_argument('pos_data', type=str, 
        help='path to positive training data', action='store')
    parser.add_argument('test_data', type=str, 
        help='path to test data', action='store')
    parser.add_argument('-pm', '--pretrained_model', dest='pretrained_model', type=str, 
        help='path to pretrained model that should be used', default='xlnet-base-cased')
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
    parser.add_argument('--split', dest='split', type=float, 
        help='define train/test split, number between 0 and 1', action='store', default=0.8)

    args = parser.parse_args()

    #torch.manual_seed(args.seed)
    #random.seed(args.seed)

    #main(args)

    main(args)


