 
import torch
import datetime

from bert.preprocess.preprocess import build_dictionary
from bert.train.train_customized import pretrain_customized

data_dir = None
train_positive_dir = 'data/signal-peptides/positive.txt'
train_negative_dir = 'data/signal-peptides/negative.txt'
val_positive_path = 'data/signal-peptides/positive.txt'
val_negative_path = 'data/signal-peptides/negative.txt'
dictionary_path = 'dic/dic.txt'
checkpoint_dir = 'checkpoint/signal-peptides/NLP+MLM/'
dataset_limit = None
epochs = 20000
batch_size = 64
print_every = 1
save_every = 5
vocabulary_size = 30000
max_len = 1024
lr = 0.001
clip_grads = 'store_true'
layers_count = 2
hidden_size = 128
heads_count = 2
d_ff = 128
dropout_prob = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = None
run_name = 'siganl-peptides-NLP+MLM-layers_count:%s-hidden_size:%s-heads_count:%s-timestamp:%s' % (\
        str(layers_count), str(hidden_size), str(heads_count),\
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
log_output = 'log/%s.log' % run_name


if __name__ == '__main__':
    #build_dictionary('data/train.txt', 'dic/dic.txt')
    pretrain_customized(data_dir, train_positive_dir, train_negative_dir, val_positive_path, val_negative_path,\
            dictionary_path, dataset_limit, vocabulary_size, batch_size, max_len, epochs,\
            clip_grads, device, layers_count, hidden_size, heads_count,\
            d_ff, dropout_prob, log_output, checkpoint_dir, print_every,\
            save_every, config
            )