import torch
from bert.train.self_training import teacher
import argparse

parser = argparse.ArgumentParser(description='teacher')
parser.add_argument('--teacher-checkpoint', '-t', dest='teacher', required=True, help='the path of teacher model')
parser.add_argument('--data-path', dest='data_path', required=True, help='the path of data file')
parser.add_argument('--batch-size', '-b', dest='batch_size', help='the batch size', default=32, type=int)
parser.add_argument('--layers-count', dest='layers_count', help='The layers count', default=2, type=int)
parser.add_argument('--hidden-size', dest='hidden_size', help='The hidden size', default=128, type=int)
parser.add_argument('--heads-count', dest='heads_count', help='The heads count', default=2, type=int)
parser.add_argument('--d-ff', dest='d_ff', help='The dff', default=128, type=int)
parser.add_argument('--dropout', dest='dropout', help='The drop out probability', default=0.1, type=float)
parser.add_argument('--run-name', dest='run_name', help='The run name', default='teacher', type=str)
parser.add_argument('--vocabulary-size', dest='vocabulary_size', help='The size of vocabulary', default=30000, type=int)
parser.add_argument('--max-len', dest='max_len', help='The maximun of input length', default=1024, type=int)
args = parser.parse_args()

teacher_checkpoint = args.teacher
data_dir = None
data_path = args.data_path
dictionary_path = args.dictionary_path
print(teacher_checkpoint)
print(data_path)
print(dictionary_path)
dataset_limit = None
batch_size = args.batch_size
vocabulary_size = args.vocabulary_size
max_len = args.max_len
layers_count = args.layers_count
hidden_size = args.hidden_size
heads_count = args.hidden_size
d_ff = args.d_ff
dropout_prob = args.dropout
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = None
run_name = '%s:%s-hidden_size:%s-heads_count:%s-timestamp:%s' % (args.run_name,\
        str(layers_count), str(hidden_size), str(heads_count),\
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
log_output = 'log/%s.log' % run_name

if __name__ == '__main__':
    teacher(data_dir, data_path, dictionary_path,\
            dataset_limit, vocabulary_size, batch_size, max_len, device,\
            layers_count, hidden_size, heads_count, d_ff, dropout_prob,\
            log_output, teacher_checkpoint, config, run_name
            )
