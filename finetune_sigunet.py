from bert.train.train_sigunet import finetuneSigunet

import torch
import datetime

pretrained_checkpoint = 'models/pretrain/NO_MASK/epoch=085-val_loss=0.00683-val_metrics=1.0-0.501.pth'
data_dir = None
train_path = 'data/finetune_features/SignalP/SignalP_euk_train.txt'
val_path = 'data/finetune_features/SignalP/SignalP_euk_train.txt'
dictionary_path = 'dic/dic.txt'
checkpoint_dir = 'models/finetune/no_concated_pre_no_msk_not_fixed_lr1e-4/'
dataset_limit = None
epochs = 1000
batch_size = 128
print_every = 1
save_every = 10
vocabulary_size = 30000
max_len = 1024
lr = 0.0001
clip_grads = 'store_true'
layers_count = 1
hidden_size = 128
heads_count = 2
d_ff = 128
dropout_prob = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = None
run_name = 'Sigunet_reproduce:%s-hidden_size:%s-heads_count:%s-timestamp:%s' % (\
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
