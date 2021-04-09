# implementing this guide: https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/

# imports
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd

# import other project files
import preprocessing as pp

# seeds
RANDOM_SEED = 2
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# hardware
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load neg data
neg_data = []
with open("./twitter-datasets/train_neg.txt") as file:
    for line in file:
        neg_data.append(line)
neg_df = pd.DataFrame(neg_data, columns = {'tweet'})
neg_df['label'] = -1
# load pos data
pos_data = []
with open("./twitter-datasets/train_pos.txt") as file:
    for line in file:
        pos_data.append(line)
pos_df = pd.DataFrame(pos_data, columns = {'tweet'})
pos_df['label'] = 1
# build mixed dataframe
df = pd.concat([neg_df, pos_df])

# checkpoint
print("Loaded Data")
print(df.shape)
print(df.head())

# apply preprocessing to tweets
df['tweet'] = df['tweet'].apply(lambda row: pp.preprocess_tweet(str(row)))
# df['tweet'] = df['tweet'].apply(lambda row: pp.spellcheck_tweet(str(row)))


# checkpoint
print("After Preprocessing Strings")
print(df.shape)
print(df.head())