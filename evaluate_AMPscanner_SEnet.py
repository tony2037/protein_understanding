import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, classification_report
from sklearn.utils import shuffle

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

from bert.train.AMP.model import AMPscanner_SEnet, AMPscanner_SEnet_poyu

import argparse
from glob import glob
from tqdm import tqdm
import os, sys

#Model params
data_dir = None
test_path = 'data/AMP/test.txt'
dictionary_path = 'dic/dic.txt'
dataset_limit = None
batch_size = 16
vocabulary_size = 30000
max_len = 1024
layers_count = 2
hidden_size = 128
heads_count = 2
d_ff = 128
dropout_prob = 0.1
device = 'cpu'

embedding_vector_length = 128
nbf = 64 	# No. Conv Filters
flen = 17 	# Conv Filter length
nlstm = 100 	# No. LSTM layers
ndrop = 0.1     # LSTM layer dropout

parser = argparse.ArgumentParser(description='AMPscanner')
parser.add_argument('checkpoint', help='checkpoint directory')
parser.add_argument('log_path', help='The file path of log file', default='AMPscanner.log')
args = parser.parse_args()
checkpoint = args.checkpoint
log_path = args.log_path
finetune_models = glob(os.path.join(checkpoint, '*.pth'))
finetune_models = [p for p in finetune_models if p.find('epoch=000') < 0]

dictionary_path = dictionary_path if data_dir is None else join(data_dir, dictionary_path)
dictionary = IndexDictionary.load(dictionary_path=dictionary_path,
                                  vocabulary_size=vocabulary_size)
vocabulary_size = len(dictionary)

test_path = test_path if data_dir is None else join(data_dir, test_path)
test_dataset = ClassificationDataset(data_path=test_path, dictionary=dictionary)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=ClassificationDataset.collate_function)
BESTS = [0., 0., 0., 0., 0., 0.]
BESTS_MODEL = ''

for finetune_model in finetune_models:
    print('=' * 35)
    print('Model: {}'.format(finetune_model))
    print('=' * 35)
    pretrained_model = build_model(layers_count, hidden_size, heads_count, d_ff, dropout_prob, max_len, vocabulary_size, forward_encoded=True)
    model = AMPscanner_SEnet_poyu(model=pretrained_model, embedding_vector_length=embedding_vector_length, nbf=nbf, flen=flen, nlstm=nlstm, ndrop=ndrop)
    state_dict = torch.load(finetune_model,  map_location=torch.device('cpu'))['state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    pred_class = []
    true_class = []
    for inputs, targets, batch_count in tqdm(test_dataloader, ncols=60):
        inputs = convert_to_tensor(inputs, device)
        targets = convert_to_tensor(targets, device)

        predictions, batch_losses = model(inputs, targets)
        predictions = convert_to_array(predictions)
        targets = convert_to_array(targets)
        predictions = predictions.squeeze()
        # predictions = np.expand_dims(predictions, axis=0)
        targets = targets.squeeze()
        # targets = np.expand_dims(targets, axis=0)
        pred_class.append(predictions)
        true_class.append(targets)

    pred_class = np.concatenate(pred_class, axis=0)
    true_class = np.concatenate(true_class, axis=0)
    print(pred_class)
    print(true_class)
    assert pred_class.shape[0] == true_class.shape[0]
    print('Total samples: {}'.format(str(pred_class.shape[0])))
    tn, fp, fn, tp = confusion_matrix(true_class, pred_class).ravel()
    roc = roc_auc_score(true_class,pred_class) * 100.0
    mcc = matthews_corrcoef(true_class,pred_class)
    acc = (tp + tn) / (tn + fp + fn + tp + 0.0) * 100.0
    sens = tp / (tp + fn + 0.0) * 100.0
    spec = tn / (tn + fp + 0.0) * 100.0
    prec = tp / (tp + fp + 0.0) * 100.0

    print("\nTP\tTN\tFP\tFN\tSens\tSpec\tAcc\tMCC\tauROC\tPrec")
    print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(tp,tn,fp,fn,np.round(sens,4),np.round(spec,4),np.round(acc,4),np.round(mcc,4),np.round(roc,4),np.round(prec,4)))

    if acc > BESTS[0] and mcc > BESTS[1]:
        BESTS[0] = acc
        BESTS[1] = mcc
        BESTS[2] = roc
        BESTS[3] = sens
        BESTS[4] = spec
        BESTS[5] = prec
        BESTS_MODEL = finetune_model

print('\nBEST model')
print("\nACC\tMCC\tROC\tsens\tspec\tprec")
print("{}\t{}\t{}\t{}\t{}\t{}".format(np.round(BESTS[0],4),np.round(BESTS[1],4),np.round(BESTS[2],4)\
                                        ,np.round(BESTS[3],4),np.round(BESTS[4],4),np.round(BESTS[5],4)))

with open(log_path, 'a+') as f:
    f.write('{}\n'.format(BESTS_MODEL))
    f.write('{}\t{}\n'.format(np.round(BESTS[0], 4), np.round(BESTS[1], 4)))
