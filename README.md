## File Structure
Some scripts may assume the following file-structure:
- Data : Directory containing all training-, test- and preprocessed data
- Models : Directory containing all fine-tuned models that we use for prediction.
- Predictions : Directory containing all predictions for test-set.
- Preprocessing_Data : Directory containing wordlists and other data used for preprocessing.
- LectureBaselines : Directory containing implementations of the baselines from the exercises. A separate ReadMe can be found here.

The following scripts should be contained in the main project folder:
- preprocessing.py : Used for preprocessing data-sets with different preprocessing methods.
- training_*.py : Training scripts for huggingface models and our ensemble models.
- inference_*.py : Scripts used to create predictions for the test set.
- utils_*.py : Contains some useful code-snippets used in the above scripts.


## Dataset

TODO: Add polybox link that leads to preprocessed datasets we use in experiments.

Download the tweet dataset:
```
wget http://www.da.inf.ethz.ch/teaching/2018/CIL/material/exercise/twitter-datasets.zip
```
Move it to the Data directory:
```
unzip twitter-datasets.zip

mkdir Data

mv twitter-datasets/* Data
```
The dataset should contain the following files:
- sample_submission.csv
- train_neg.txt : a subset of negative training samples
- train_pos.txt: a subset of positive training samples
- test_data.txt:
- train_neg_full.txt: the full negative training samples
- train_pos_full.txt: the full positive training samples


## Additional Datasets
Download the tweet datasets:
```
wget [download link]
```

Match the format of the original dataset and create a positive and negative datasets (save them in Data/):
- For the dataset from [Sentiment140](https://www.kaggle.com/kazanova/sentiment140?select=training.1600000.processed.noemoticon.csv)
```
python3 additional_dataset_1.py dataset1.csv

```
- For the dataset from [Tweet Sentiment Extraction](https://www.kaggle.com/c/tweet-sentiment-extraction/data?select=train.csv)

```
python3 additional_dataset_2.py dataset2.csv

```
Combine all datasets:
- for negative:
```
python3 combine_datasets.py Data/train_neg_full.txt Data/train_neg_add1.txt Data/train_neg_add2.txt train_neg_all_full.txt

```
- for positive:
```
python3 combine_datasets.py Data/train_pos_full.txt Data/train_pos_add1.txt Data/train_pos_add2.txt train_pos_all_full.txt

```


## General Workflow

### Preprocessing
```
python3 preprocessing.py Data/train_pos_full.txt Data/train_pos_full_basic.txt -v

python3 preprocessing.py Data/train_neg_full.txt Data/train_neg_full_basic.txt -v

python3 preprocessing.py Data/train_pos_all_full.txt Data/train_pos_all_full_basic.txt -v

python3 preprocessing.py Data/train_neg_all_full.txt Data/train_neg_all_full_basic.txt -v

python3 preprocessing.py Data/train_pos_full.txt Data/train_pos_full_augmented.txt -v -a

python3 preprocessing.py Data/train_neg_full.txt Data/train_neg_full_augmented.txt -v -a

python3 preprocessing.py Data/train_pos_full.txt Data/train_pos_full_pp.txt -v -s -l -sw

python3 preprocessing.py Data/train_neg_full.txt Data/train_neg_full_pp.txt -v -s -l -sw
```
### Training
```
bsub -W 24:00 -R "rusage[mem=8192]" -R "rusage[ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" -oo Output/BART_raw.out 
```
raw-data
```
python3 training_huggingface.py -v -c facebook/bart-base -e 3 -bs 8 -as 4 Data/train_neg_full.txt Data/train_pos_full.txt Models/bart_raw

python3 training_huggingface.py -v -c bert-base-uncased -e 3 -bs 8 -as 4 Data/train_neg_full.txt Data/train_pos_full.txt Models/bert_raw

python3 training_huggingface.py -v -c vinai/bertweet-base -e 3 -bs 8 -as 4 Data/train_neg_full.txt Data/train_pos_full.txt Models/bertweet_raw

python3 training_huggingface.py -v -c xlnet-base-cased -e 3 -bs 8 -as 4 Data/train_neg_full.txt Data/train_pos_full.txt Models/xlnet_raw
```
preprocessed-data
```
python3 training_huggingface.py -v -c xlnet-base-cased -e 3 -bs 8 -as 4 Data/train_neg_full_pp.txt Data/train_pos_full_pp.txt Models/xlnet_pp

python3 training_huggingface.py -v -c bert-base-uncased -e 3 -bs 8 -as 4 Data/train_neg_full_pp.txt Data/train_pos_full_pp.txt Models/bert_pp

python3 training_huggingface.py -v -c facebook/bart-base -e 3 -bs 8 -as 4 Data/train_neg_full_pp.txt Data/train_pos_full_pp.txt Models/bart_pp

python3 training_huggingface.py -v -c vinai/bertweet-base -e 3 -bs 8 -as 4 Data/train_neg_full_pp.txt Data/train_pos_full_pp.txt Models/bertweet_pp
```
augmented-data
```
python3 training_huggingface.py -v -c bert-base-uncased -e 3 -bs 8 -as 4 Data/train_neg_full_augmented.txt Data/train_pos_full_augmented.txt Models/bert_augmented

python3 training_huggingface.py -v -c facebook/bart-base -e 3 -bs 8 -as 4 Data/train_neg_full_augmented.txt Data/train_pos_full_augmented.txt Models/bart_augmented

python3 training_huggingface.py -v -c xlnet-base-cased -e 3 -bs 8 -as 4 Data/train_neg_full_augmented.txt Data/train_pos_full_augmented.txt Models/xlnet_augmented

python3 training_huggingface.py -v -c vinai/bertweet-base -e 3 -bs 8 -as 4 Data/train_neg_full_augmented.txt Data/train_pos_full_augmented.txt Models/bertweet_augmented
```
additional-data
```
python3 training_huggingface.py -v -c bert-base-uncased -e 3 -bs 8 -as 4 Data/train_neg_all_full.txt Data/train_pos_all_full.txt Models/bert_all

python3 training_huggingface.py -v -c facebook/bart-base -e 3 -bs 8 -as 4 Data/train_neg_all_full.txt Data/train_pos_all_full.txt Models/bart_all

python3 training_huggingface.py -v -c vinai/bertweet-base -e 3 -bs 8 -as 4 Data/train_neg_all_full.txt Data/train_pos_all_full.txt Models/bertweet_all

python3 training_huggingface.py -v -c xlnet-base-cased -e 3 -bs 8 -as 4 Data/train_neg_all_full.txt Data/train_pos_all_full.txt Models/xlnet_all
```

Create predictions for test-data:
```
python3 inference.py Data/test_data_basic.txt Predictions/bart-base_basic.csv -dt Pretrained_Models/bart-base/ -dm Models/bart-base_basic/checkpoint_3 

python3 inference.py Data/test_data_basic.txt Predictions/bert-base-uncased_basic.csv -dt Pretrained_Models/bert-base-uncased/ -dm Models/bert-base-uncased_basic/checkpoint_3 

python3 inference.py Data/test_data_basic.txt Predictions/xlnet-base-cased_basic.csv -dt Pretrained_Models/xlnet-base-cased/ -dm Models/xlnet-base-cased_basic/checkpoint_3 -x
```

## Virtual Environment & Dependencies

Start virtual environment:
```bash
source venv/bin/activate
```

Exit virtual environment:
```
deactivate
```
Update requirements.txt:
```
pip list --format=freeze > requirements.txt
```
Install dependencies (make sure to be in venv):
```bash
pip install -r requirements.txt
```

## Leonhard Cluster

Load modules:
```
module load gcc/6.3.0 python_gpu/3.8.5 eth_proxy
```
Reset modules:
```
module purge
module load StdEnv
```
Submitting job:
```
bsub -R "rusage[mem=8192]" -R "rusage[ngpus_excl_p=1]" -oo output python3 main.py [args]
```
Submitting as interactive job for testing (output to terminal):
```
bsub -I -R "rusage[mem=8192]" -R "rusage[ngpus_excl_p=1]" python3 main.py [args]
```
Monitoring job:
```
bbjobs
```