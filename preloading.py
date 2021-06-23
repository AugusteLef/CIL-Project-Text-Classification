# loading bert
from transformers import BertTokenizer, BertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer.save_pretrained('Pretrained/bert-base-uncased')
model.save_pretrained('Pretrained/bert-base-uncased')

# loading bart
from transformers import BartTokenizer, BartForSequenceClassification
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForSequenceClassification.from_pretrained('facebook/bart-base')
tokenizer.save_pretrained('Pretrained/bart-base')
model.save_pretrained('Pretrained/bart-base')

# loading XLNET
from transformers import XLNetTokenizer,XLNetForSequenceClassification
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased')
tokenizer = XLNetTokenize.from_pretrained('xlnet-base-cased')
tokenizer.save_pretrained('Pretrained/xlnet-base-cased')
model.save_pretrained('Pretrained/xlnet-base-cased')