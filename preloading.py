# loading bert
print("Loading bert-base-uncased...")
from transformers import BertTokenizer, BertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer.save_pretrained('Pretrained_Models/bert-base-uncased')
model.save_pretrained('Pretrained_Models/bert-base-uncased')

# loading bart
print("Loading bart-base...")
from transformers import BartTokenizer, BartForSequenceClassification
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForSequenceClassification.from_pretrained('facebook/bart-base')
tokenizer.save_pretrained('Pretrained_Models/bart-base')
model.save_pretrained('Pretrained_Models/bart-base')

# loading XLNET
print("Loading xlnet-base-cased...")
from transformers import XLNetTokenizer, XLNetForSequenceClassification
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased')
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
tokenizer.save_pretrained('Pretrained_Models/xlnet-base-cased')
model.save_pretrained('Pretrained_Models/xlnet-base-cased')

# nltk data
print("Loading nltk data...")
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# slang json
print("Loading myslang.json...")
import requests
import json
import bs4
resp = requests.get("http://www.netlingo.com/acronyms.php")
soup = bs4.BeautifulSoup(resp.text, "html.parser")
slangdict = {}
key = ""
value = ""
for div in soup.findAll('div', attrs={'class': 'list_box3'}):
    for li in div.findAll('li'):
        for a in li.findAll('a'):
            key = a.text
            value = li.text.split(key)[1]
            slangdict[key] = value
# store in json format
with open('Preprocessing_Data/myslang.json', 'w') as f:
    json.dump(slangdict, f, indent=2)