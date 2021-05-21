import torch

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.labels != None:
            return self.texts[idx], self.labels[idx]
        else:
            return (self.texts[idx],)

class TextCollator():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, list_items):
        texts = [item[0] for item in list_items]
        batch = self.tokenizer(texts, truncation=True, padding=True)
        if 1 < len(list_items[0]):
            labels = [item[1] for item in list_items]
            batch["labels"] = labels
        batch = {key: torch.tensor(val) for key, val in batch.items()}
        return batch

