import torch
from transformers import BertTokenizer, BertForSequenceClassification

PATH_DATA_NEG = "Data/train_neg_preprocessed.txt"
PATH_DATA_POS = "Data/train_pos_preprocessed.txt"
PRE_TRAINED_MODEL_NAME = "bert-base-uncased"
MAX_SEQ_LEN = 98
N_EPOCHS = 1
BATCH_SIZE = 1

class TwitterDataset(torch.utils.data.Dataset):

    def __init__(self, tweets, labels, tokenizer, max_len):
        """
        TODO: comment
        """
        super().__init__()
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """
        :return: length of the dataset
        """
        return len(self.tweets)

    def __getitem__(self, idx):
        """
        :param idx: index
        :return: item at index idx
        """
        tweet = self.tweets[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        return input_ids, attention_mask, label

class BertBinarySeqClassifier(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.bert_model = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.softmax = torch.nn.Softmax()
    
    def forward(self, input_ids, attention_mask):
        tmp = self.bert_model(input_ids, attention_mask)
        return self.softmax(tmp[0])

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f_neg = open(PATH_DATA_NEG)
    l_neg = f_neg.readlines()
    f_pos = open(PATH_DATA_POS)
    l_pos = f_pos.readlines()
    l = l_neg + l_pos
    labels = [0] * len(l_neg) + [1] * len(l_pos)

    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME) 

    ds = TwitterDataset(l, labels, tokenizer, MAX_SEQ_LEN) 
    dl = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    
    model = BertBinarySeqClassifier()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(N_EPOCHS):
        for input_ids, attention_mask, label in dl:
            input_ids = input_ids.squeeze(dim=0)
            attention_mask = attention_mask.squeeze(dim=0)
            pred = model(input_ids, attention_mask)
            pred.unsqueeze(dim=0)
            loss = loss_fn(pred, label)
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

if __name__ == "__main__":
    main()

