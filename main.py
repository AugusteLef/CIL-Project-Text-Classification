import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from pathlib import Path

DIR_DATA = "Data"
PATH_DATA_NEG = os.path.join(DIR_DATA, "train_neg_preprocessed.txt")
PATH_DATA_POS = os.path.join(DIR_DATA, "train_pos_preprocessed.txt")
DIR_PRETRAINED = "Pretrained/bert-base-uncased"
N_EPOCHS = 3
BATCH_SIZE = 16
ACCUMULATION_SIZE = 16
VERBOSE = True

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

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir == "neg" else 1)

    return texts, labels

def main():

    if VERBOSE: print("reading data...")
    #f_neg = open(PATH_DATA_NEG)
    #texts_neg = f_neg.readlines()
    #f_pos = open(PATH_DATA_POS)
    #texts_pos = f_pos.readlines()
    #texts = texts_neg + texts_pos
    #labels = [0] * len(texts_neg) + [1] * len(texts_pos)
    texts, labels = read_imdb_split(os.path.join(DIR_DATA, 'aclImdb/train'))

    if VERBOSE: print("loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(DIR_PRETRAINED) 

    if VERBOSE: print("tokenizing data...")
    encodings = tokenizer(texts, truncation=True, padding=True)
    
    ds = EncodingsDataset(encodings, labels) 
    dl = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=int(BATCH_SIZE/ACCUMULATION_SIZE),
        shuffle=True,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BertForSequenceClassification.from_pretrained(DIR_PRETRAINED)
    model = torch.nn.DataParallel(model)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)

    if VERBOSE: print("training...")
    model.train()
    for epoch in range(N_EPOCHS):
        if VERBOSE: print("epoch %d..." % epoch)
        avg_loss = torch.tensor(0.0)
        for i, batch in enumerate(dl):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss /= ACCUMULATION_SIZE
            loss.backward()
            avg_loss += loss.detach()
            if (i + 1) % ACCUMULATION_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad() 
                print(avg_loss)
                avg_loss = torch.tensor(0.0)

if __name__ == "__main__":
    main()

