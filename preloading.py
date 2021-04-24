from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer.save_pretrained('Pretrained/bert-base-uncased')
model.save_pretrained('Pretrained/bert-base-uncased')

