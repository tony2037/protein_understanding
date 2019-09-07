
import torch
import datetime

from bert.preprocess.preprocess import build_dictionary
from bert.train.train import pretrain

data_dir = None
train_dir = 'data/reconstructed_pretrain/reconstructed_pretrain.txt'
val_path = 'data/reconstructed_pretrain/reconstructed_pretrain.txt'
dictionary_path = 'dic/dic.txt'
checkpoint_dir = 'data/reconstructed_pretrain/checkpoint_0.15_original'
dataset_limit = None
epochs = 100
batch_size = 16
print_every = 1
save_every = 5
vocabulary_size = 100
max_len = 1024
lr = 0.001
clip_grads = 'store_true'
layers_count = 1
hidden_size = 128
heads_count = 2
d_ff = 128
dropout_prob = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = None
run_name = 'BERT-layers_count:%s-hidden_size:%s-heads_count:%s-timestamp:%s' % (\
        str(layers_count), str(hidden_size), str(heads_count),\
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
log_output = 'log/%s.log' % run_name


if __name__ == '__main__':
    build_dictionary('data/train.txt', 'dic/dic.txt')
    pretrain(data_dir, train_dir, val_path, dictionary_path,\
            dataset_limit, vocabulary_size, batch_size, max_len, epochs,\
            clip_grads, device, layers_count, hidden_size, heads_count,\
            d_ff, dropout_prob, log_output, checkpoint_dir, print_every,\
            save_every, config
            )
