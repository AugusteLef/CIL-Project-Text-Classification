import numpy as np
import pandas as pd
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
from torch.utils.data import TensorDataset,DataLoader,RandomSampler,SequentialSampler



#global variables
POSITIVE_PATH = '/cluster/home/alefevre/data/twitter/Datas/train_pos.txt'
NEGATIVE_PATH  = '/cluster/home/alefevre/data/twitter/Datas/train_neg.txt'
OUTPUT_PREDS_PATH = '/cluster/home/alefevre/results/twitter/XLNET_test_preds.txt'
N_EPOCHS = 5 # or 3 ?
BATCH_SIZE = 16

def preprocess_tweet(tweet):
    '''
    :param tweet: tweet as a string
    :return: string s which is preprocessed tweet
    '''
    tweet = re.sub('<user>', '', tweet) # remove user tags
    tweet = re.sub('<url>', '', tweet) # remove url tags
    tweet = re.sub('#\w*', '', tweet) # remove hashtags
    tweet = re.sub('[0-9]', '', tweet) # remove numbers
    tweet = re.sub('[^\w\s]', '', tweet) # remove punctuation
    tweet = re.sub('\s+', ' ', tweet) # remove excess whitespace
    tweet = re.sub('^\s', '', tweet) # remove excess whitespace
    tweet = re.sub('\s$', '', tweet) # remove excess whitespace
    tweet += '\n'
    #tweet = ' '.join(ws.segment(tweet)) # segment words
    #tweet = ' '.join([speller(w) for w in tweet.split()]) # spell checking
    return tweet

def load_tweet(path, label):
    data = []
    with open(path) as file:
        for line in file:
            data.append(line)
    data_df = pd.DataFrame(data, columns = {'tweet'})
    data_df['label'] = label
    print('Loaded data: ' + path)
    return data_df

def concat_data(dset1, dset2):
    df = pd.concat([dset1, dset2])
    print('Concatenated data')
    return df

#XLNET need sep and cls tags at the end of each tweet
def XLNET_tweets_transformation(df):
    tweets = []
    for tweet in df['tweet']:
        tweet = tweet+'[SEP] [CLS]'
        tweets.append(tweet)
    return tweets

def get_max_length_ids(ids):
    maxlen = len(ids[0])
    for i in ids:
        if (len(i)>maxlen):
            maxlen = len(i)
    return maxlen

def accuracy(preds, labels):
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
    #LOAD AND BASIC PP

    #global variables
    POSITIVE_PATH = '/cluster/home/alefevre/data/twitter/Datas/train_pos.txt'
    NEGATIVE_PATH  = '/cluster/home/alefevre/data/twitter/Datas/train_neg.txt'

    #load datas
    pos_df = load_tweet(POSITIVE_PATH, 1)
    neg_df = load_tweet(NEGATIVE_PATH, 0)

    #concat datas
    all_tweet_df = concat_data(pos_df, neg_df)

    #apply basic preprocessing (tobias)
    all_tweet_df['tweet'] = all_tweet_df['tweet'].apply(lambda row: preprocess_tweet(str(row)))

    #XLNET pp
    tweets = XLNET_tweets_transformation(all_tweet_df)

    #Tokenization
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased',do_lower_case=True)
    tokenized_tweets = [tokenizer.tokenize(tweet) for tweet in tweets]

    #ids and label
    ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_tweets]
    labels = all_tweet_df['label'].values

    #get max length and pad ids
    MAX_LENGTH = get_max_length_ids(ids)
    padded_ids = pad_sequences(ids, maxlen=MAX_LENGTH, dtype="long", truncating="post", padding="post")

    #splitdataset, test size = 10% 
    xtrain,xtest,ytrain,ytest = train_test_split(padded_ids, labels, test_size=0.10)

    

    #transform dataset to torch.tensor format
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
    model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased')

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

        #testing
    model.eval()
    acc = []
    lab = []
    t = 0
    for inp,lab1 in test_loader:
        inp.to(device)
        lab1.to(device)
        t+=lab1.size(0)
        outp1 = model(inp.to(device))
        [acc.append(p1.item()) for p1 in torch.argmax(outp1[0],axis=1).flatten() ]
        [lab.append(z1.item()) for z1 in lab1]
    print("Total Examples : {} Accuracy {}".format(t,accuracy(acc,lab)))

    to_save = []
    to_save.append(acc)
    to_save.append(lab)
    to_save.append([accuracy(acc,lab)])
    np.savetxt(OUTPUT_PREDS_PATH, np.asarray(to_save), delimiter=',', fmt='%s')

main()


