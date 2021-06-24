# loading bert
from transformers import BertTokenizer, BertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer.save_pretrained('Pretrained_Models/bert-base-uncased')
model.save_pretrained('Pretrained_Models/bert-base-uncased')

# loading bart
from transformers import BartTokenizer, BartForSequenceClassification
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForSequenceClassification.from_pretrained('facebook/bart-base')
tokenizer.save_pretrained('Pretrained_Models/bart-base')
model.save_pretrained('Pretrained_Models/bart-base')

# loading XLNET
from transformers import XLNetTokenizer, XLNetForSequenceClassification
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased')
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
tokenizer.save_pretrained('Pretrained_Models/xlnet-base-cased')
model.save_pretrained('Pretrained_Models/xlnet-base-cased')