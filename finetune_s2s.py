from bert.train.train_s2s import finetuneSeq2Seq

import torch
import datetime

pretrained_checkpoint = 'data/seq2seq/epoch=100-val_loss=2.71-val_metrics=0.183-0.5.pth'
data_dir = None
train_path = 'data/seq2seq/example.txt'
val_path = 'data/seq2seq/example.txt'
dictionary_path = 'dic/dic.txt'
checkpoint_dir = 'data/seq2seq/FineTune'
dataset_limit = None
epochs = 10
batch_size = 16
print_every = 1
save_every = 10
vocabulary_size = 30000
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
fixed_length = None
fixed_length = 600

if __name__ == '__main__':
    finetuneSeq2Seq(pretrained_checkpoint,\
            data_dir, train_path, val_path, dictionary_path,\
            vocabulary_size, batch_size, max_len, epochs,\
            lr, clip_grads, device, layers_count, hidden_size, heads_count,\
            d_ff, dropout_prob, log_output, checkpoint_dir, print_every,\
            save_every, config, fixed_length=fixed_length
            )
