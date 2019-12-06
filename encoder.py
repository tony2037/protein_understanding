import numpy as np

import torch 
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F

from bert.preprocess import PAD_INDEX
from bert.preprocess.dictionary import IndexDictionary
from bert.train.model.bert import build_model
from bert.train.utils.stateload import stateLoading
from bert.train.utils.fix_weights import disable_grad
from bert.train.utils.convert import convert_to_tensor, convert_to_array
from bert.train.datasets.NoOneHot import ClassificationDataset

import argparse
from glob import glob
from tqdm import tqdm
import os, sys

import numpy as np
from keras.utils import to_categorical

parser = argparse.ArgumentParser(description='Encoder')
parser.add_argument('model_path', help='the path of pretrain bert model')
parser.add_argument('input', help='the path of input file')
parser.add_argument('output', help='The path of output file', default='encoded.np')
parser.add_argument('seq_len', help='The sequence length', default=200)
parser.add_argument('--dictionary-path', dest='dictionary_path', help='the path of dictionary file', type=str, default='dic/dic.txt')
parser.add_argument('--batch-size', '-b', dest='batch_size', help='the batch size', default=1, type=int)
parser.add_argument('--layers-count', dest='layers_count', help='The layers count', default=2, type=int)
parser.add_argument('--hidden-size', dest='hidden_size', help='The hidden size', default=128, type=int)
parser.add_argument('--heads-count', dest='heads_count', help='The heads count', default=2, type=int)
parser.add_argument('--d-ff', dest='d_ff', help='The dff', default=128, type=int)
parser.add_argument('--dropout', dest='dropout', help='The drop out probability', default=0.1, type=float)
parser.add_argument('--run-name', dest='run_name', help='The run name', default='teacher', type=str)
parser.add_argument('--vocabulary-size', dest='vocabulary_size', help='The size of vocabulary', default=30000, type=int)
parser.add_argument('--max-len', dest='max_len', help='The maximun of input length', default=513, type=int)
parser.add_argument('--cat-onehot', dest='onehot', help='If gonna concatenate one-hot', default=False, type=bool)
args = parser.parse_args()
checkpoint = args.model_path
output = args.output
input_file = args.input
seq_len = int(args.seq_len)

data_dir = None
dictionary_path = args.dictionary_path
dataset_limit = None
batch_size = args.batch_size
max_len = args.max_len
layers_count = args.layers_count
hidden_size = args.hidden_size
heads_count = args.heads_count
d_ff = args.d_ff
dropout_prob = args.dropout
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = None

cat_onehot = args.onehot

dictionary_path = dictionary_path if data_dir is None else join(data_dir, dictionary_path)
dictionary = IndexDictionary.load(dictionary_path=dictionary_path,
                                  vocabulary_size=3000)
vocabulary_size = len(dictionary)

input_file = input_file if data_dir is None else join(data_dir, input_file)
dataset = ClassificationDataset(data_path=input_file, dictionary=dictionary)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=ClassificationDataset.collate_function)

model = build_model(layers_count, hidden_size, heads_count, d_ff, dropout_prob, max_len, vocabulary_size, forward_encoded=True)
# state_dict = torch.load(finetune_model,  map_location=torch.device('cpu'))['state_dict']

# model.load_state_dict(state_dict)
model = stateLoading(model, checkpoint)
model = disable_grad(model)
model.eval()
model.to(device=device)

results = []
for inputs, targets, batch_count in tqdm(dataloader, ncols=60):
    inputs = convert_to_tensor(inputs, device)
    """
    if cat_onehot:
        cache = []
        for i in inputs[0]:
            cache.append(to_categorical(i))
    onehots = np.concatenate(cache, axis=0)
    print(onehots.shape)
    exit()
    """
    tmp = [[dictionary.index_to_token(j.item()) for j in i] for i in inputs[0]]

    _, _, encoded_sources = model(inputs)
    encoded_sources = convert_to_array(encoded_sources)
    cache = []
    inputs = convert_to_array(inputs)
    for i, encoded_source in zip(inputs[0], encoded_sources):
        cutting = np.where(i == PAD_INDEX)
        if cutting[0].size > 0:
            cutting = cutting[0][0]
            cutten_source = encoded_source[:cutting]
        else:
            cutten_source = encoded_source[:]

        l, e = cutten_source.shape
        padded_len = seq_len - l
        zeros = np.zeros((padded_len, e))
        cutten_source = np.concatenate([cutten_source, zeros], axis=0)
        if cat_onehot:
            i_length = i.size
            i_padded = np.concatenate((i, np.zeros(seq_len - i_length)))
            onehot = to_categorical(i_padded, num_classes=vocabulary_size)
            cutten_source = np.concatenate((cutten_source, onehot), axis=1)
        cache.append(np.expand_dims(cutten_source, axis=0))
    encoded_sources = np.concatenate(cache, axis=0)
    results.append(encoded_sources)

    """
    b, l, e = encoded_sources.shape
    padded_len = seq_len - l
    zeros = np.zeros((b, padded_len, e))
    encoded_sources = np.concatenate([encoded_sources, zeros], axis=1)
    results.append(encoded_sources)
    """
    for t in tmp:
        print('=' * 50)
        print(' '.join(t).replace('[PAD]', ''))
        print('=' * 50)

results = np.concatenate(results, axis=0)
print(results[-1][3])
print(results[-1][4])
print('Save at %s' % output)
np.save(output, results)
