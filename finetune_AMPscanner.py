from bert.train.train_AMPscanner import finetuneAMPscanner

import torch
import datetime

pretrained_checkpoint = 'checkpoint/AMP/Roberta/epoch=485-val_loss=1.78-val_metrics=0.47-0.41.pth'
data_dir = None
train_path = 'data/AMP/train.txt'
val_path = 'data/AMP/val.txt'
dictionary_path = 'dic/dic.txt'
checkpoint_dir = 'data/AMP/AMP_1.78/checkpoint'
dataset_limit = None
epochs = 50
batch_size = 32
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
run_name = 'AMPscanner:%s-hidden_size:%s-heads_count:%s-timestamp:%s' % (\
        str(layers_count), str(hidden_size), str(heads_count),\
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
log_output = 'log/%s.log' % run_name

embedding_vector_length = 128
nbf = 64
flen = 17
nlstm = 100
ndrop = 0.1

if __name__ == '__main__':
    finetuneAMPscanner(pretrained_checkpoint,\
            data_dir, train_path, val_path, dictionary_path,\
            vocabulary_size, batch_size, max_len, epochs,\
            lr, clip_grads, device, layers_count, hidden_size, heads_count,\
            d_ff, dropout_prob, log_output, checkpoint_dir, print_every,\
            save_every, embedding_vector_length, nbf, flen, nlstm, ndrop, config
            )