import numpy as np

import torch 
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F

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

parser = argparse.ArgumentParser(description='Encoder')
parser.add_argument('model_path', help='the path of pretrain bert model')
parser.add_argument('input', help='the path of input file')
parser.add_argument('output', help='The path of output file', default='encoded.np')
parser.add_argument('seq_len', help='The sequence length', default=200)
args = parser.parse_args()
checkpoint = args.model_path
output = args.output
input_file = args.input
seq_len = int(args.seq_len)

data_dir = None
dictionary_path = 'dic/dic.txt'
dataset_limit = None
batch_size = 1
max_len = 1024
layers_count = 2
hidden_size = 128
heads_count = 2
d_ff = 128
dropout_prob = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = None

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
    tmp = [[dictionary.index_to_token(j.item()) for j in i] for i in inputs[0]]

    _, _, encoded_sources = model(inputs)
    encoded_sources = convert_to_array(encoded_sources)
    b, l, e = encoded_sources.shape
    padded_len = seq_len - l
    zeros = np.zeros((b, padded_len, e))
    encoded_sources = np.concatenate([encoded_sources, zeros], axis=1)
    results.append(encoded_sources)
    for t in tmp:
        print('=' * 50)
        print(' '.join(t).replace('[PAD]', ''))
        print('=' * 50)

results = np.concatenate(results, axis=0)
np.save(output, results)
