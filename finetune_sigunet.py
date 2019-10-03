from bert.train.train_sigunet import finetuneSigunet

import torch
import datetime

pretrained_checkpoint = 'data/signal-peptides/checkpoint/SignalP_euk_pretrain/epoch=1190-val_loss=1.74-val_metrics=0.485-0.272.pth'
data_dir = None
train_path = 'data/signal-peptides/SignalP_train_euk_96_res_label.txt'
val_path = 'data/signal-peptides/SignalP_val_euk_96_res_label.txt'
dictionary_path = 'dic/dic.txt'
checkpoint_dir = 'data/signal-peptides/checkpoint/[0.48][fix][no_concat]'
dataset_limit = None
epochs = 500
batch_size = 128
print_every = 1
save_every = 10
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
run_name = '[0.48][fix][no_concat]:%s-hidden_size:%s-heads_count:%s-timestamp:%s' % (\
        str(layers_count), str(hidden_size), str(heads_count),\
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
log_output = 'log/%s.log' % run_name

if __name__ == '__main__':
    finetuneSigunet(pretrained_checkpoint,\
            data_dir, train_path, val_path, dictionary_path,\
            vocabulary_size, batch_size, max_len, epochs,\
            lr, clip_grads, device, layers_count, hidden_size, heads_count,\
            d_ff, dropout_prob, log_output, checkpoint_dir, print_every,\
            save_every, config
            )
