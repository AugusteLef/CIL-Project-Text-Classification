import readTweets

global default_path
global train_pos_path
global train_pos_full_path
global train_neg_path
global train_neg_full_path

default_path = '/Volumes/MarcWatineHD/ETH/CIL/project_data/twitter-datasets/'
train_pos_path = default_path + 'train_pos.txt'
train_pos_full_path = default_path + 'train_pos_full.txt'
train_neg_path = default_path + 'train_neg.txt'
train_neg_full_path = default_path + 'train_neg_full.txt'

def get_cleaned_tweets(tweets_data):
    print('in get_cleaned_tweets')
    cleaned = [None] * len(tweets_data)
    print(len(tweets_data))
    for i in range(len(tweets_data)):
        print(tweets_data[i])
        cleaned[i] = readTweets.preprocess(tweets_data[i])
        print('finished'+ str(i))
    return cleaned

if __name__ == '__main__':
    train_pos_data = readTweets.read_tweets(train_pos_path)
    train_neg_data = readTweets.read_tweets(train_neg_path)
    train_pos_clean = get_cleaned_tweets(train_pos_data[:5])
    train_neg_clean = get_cleaned_tweets(train_neg_data[:5])
    print(train_neg_clean[0])

    print(train_pos_clean)
    with open('train_pos_clean_1000.txt', 'w') as f:
        f.write(str(train_pos_clean))
    with open('train_neg_clean_1000.txt', 'w') as f:
        f.write(str(train_neg_clean))
    print("hi")