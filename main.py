 
import torch
import datetime
import argparse

from bert.preprocess.preprocess import build_dictionary
from bert.train.train import pretrain

parser = argparse.ArgumentParser(description='Pretrain')
parser.add_argument('--train-dir', dest='train_dir', required=True, help='the path of training dataset')
parser.add_argument('--val-path', dest='val_path', required=True, help='the path of validation dataset')
parser.add_argument('--checkpoint-dir', dest='checkpoint_dir', help='the path of validation dataset',type=str, default='tmp/checkpoint')
parser.add_argument('--epochs', '-e', dest='epochs', help='Training epochs', type=int, default=500)
parser.add_argument('--dictionary-path', dest='dictionary_path', help='the path of dictionary file', type=str, default='dic/dic.txt')
parser.add_argument('--batch-size', '-b', dest='batch_size', help='the batch size', default=64, type=int)
parser.add_argument('--save-every', '-s', dest='save_every', help='How many epochs your take to save a model', type=int, default=5)
parser.add_argument('--layers-count', dest='layers_count', help='The layers count', default=2, type=int)
parser.add_argument('--hidden-size', dest='hidden_size', help='The hidden size', default=128, type=int)
parser.add_argument('--heads-count', dest='heads_count', help='The heads count', default=2, type=int)
parser.add_argument('--d-ff', dest='d_ff', help='The dff', default=128, type=int)
parser.add_argument('--dropout', dest='dropout', help='The drop out probability', default=0.1, type=float)
parser.add_argument('--run-name', dest='run_name', help='The run name', default='teacher', type=str)
parser.add_argument('--vocabulary-size', dest='vocabulary_size', help='The size of vocabulary', default=30000, type=int)
parser.add_argument('--max-len', dest='max_len', help='The maximun of input length', default=1024, type=int)
parser.add_argument('--learning-rate', '-lr', dest='lr', help='Learning rate', type=float, default=0.001)
parser.add_argument('--Pretrained-model', dest='pretrained_model', help='The path of pretrained model', default=None)
args = parser.parse_args()

data_dir = None
# train_dir = 'data/AMP/AMP.total_1121.1e-3.pretrain.txt'
train_dir = args.train_dir
val_path = args.val_path
dictionary_path = args.dictionary_path
checkpoint_dir = args.checkpoint_dir
dataset_limit = None
epochs = args.epochs
batch_size = args.batch_size
print_every = 1
save_every = args.save_every
vocabulary_size = args.vocabulary_size
max_len = args.max_len
lr = args.lr
clip_grads = 'store_true'
layers_count = args.layers_count
hidden_size = args.hidden_size
heads_count = args.heads_count
d_ff = args.d_ff
dropout_prob = args.dropout
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = None
run_name = 'Pretraining-layers_count:%s-hidden_size:%s-heads_count:%s-timestamp:%s' % (\
        str(layers_count), str(hidden_size), str(heads_count),\
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
log_output = 'log/%s.log' % run_name
pretrained_model = args.pretrained_model


if __name__ == '__main__':
    #build_dictionary('data/train.txt', 'dic/dic.txt')
    pretrain(data_dir, train_dir, val_path, dictionary_path,\
            dataset_limit, vocabulary_size, batch_size, max_len, epochs,\
            clip_grads, device, layers_count, hidden_size, heads_count,\
            d_ff, dropout_prob, log_output, checkpoint_dir, print_every,\
            save_every, pretrained_model, config
            )
