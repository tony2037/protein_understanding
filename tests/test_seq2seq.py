import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append(os.getcwd())

from torch.utils.data import DataLoader
from tqdm import tqdm

from bert.preprocess.dictionary import IndexDictionary
from bert.train.utils.convert import convert_to_tensor

from bert.train.datasets.seq2seq import Seq2SeqDataset
from bert.train.utils.collate import seq2seq_collate_function

dictionary = IndexDictionary.load(dictionary_path='dic/dic.txt', vocabulary_size=100)
vocabulary_size = len(dictionary)
dataset = Seq2SeqDataset('data/seq2seq/example.txt', dictionary)
dataloader = DataLoader(dataset, batch_size=16, collate_fn=seq2seq_collate_function)

for inputs, targets, batch_count in tqdm(dataloader):
    inputs = convert_to_tensor(inputs, None)
    targets = convert_to_tensor(targets, None)
    assert inputs[0].shape[0] == inputs[1].shape[0] == targets.shape[0]
    assert inputs[0].shape[1] == inputs[1].shape[1] == targets.shape[1]
