from bert.preprocess.dictionary import IndexDictionary
from bert.preprocess.mutation_matrix import MutationMatrix
from bert.preprocess import PAD_TOKEN, UNK_TOKEN, MASK_TOKEN, CLS_TOKEN, SEP_TOKEN
from .model.bert import build_model, FineTuneModel
from .loss_models import MLMNSPLossModel, ClassificationLossModel, MLMLossModel, MutationMLMLossModel
from .metrics import mlm_accuracy, nsp_accuracy, classification_accuracy
from .datasets.pretraining import PairedDataset
from .datasets.classification import SST2IndexedDataset
from .utils.stateload import stateLoading
from .utils.fix_weights import disable_grad
from .trainer import Trainer
from .utils.log import make_run_name, make_logger, make_checkpoint_dir
from .utils.collate import pretraining_collate_function, classification_collate_function
from .utils.stateload import stateLoading
from .utils.convert import convert_to_tensor, convert_to_array
from .optimizers import NoamOptimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
from torch.nn import DataParallel
import torch.nn.functional as F
from torch.utils.data import DataLoader

import random
import numpy as np
from os.path import join
from tqdm import tqdm

RUN_NAME_FORMAT = (
    "BERT-"
    "{phase}-"
    "layers_count={layers_count}-"
    "hidden_size={hidden_size}-"
    "heads_count={heads_count}-"
    "{timestamp}"
)

def teacher(data_dir, data_path, dictionary_path,
             dataset_limit, vocabulary_size, batch_size, max_len, device,
             layers_count, hidden_size, heads_count, d_ff, dropout_prob,
             log_output, checkpoint, config, run_name=None, confidence=0.4, pseudo_data_path='pseudo_data.txt', **_):

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    data_path = data_path if data_dir is None else join(data_dir, data_path)
    dictionary_path = dictionary_path if data_dir is None else join(data_dir, dictionary_path)

    run_name = run_name if run_name is not None else make_run_name(RUN_NAME_FORMAT, phase='teacher', config=config)
    logger = make_logger(run_name, log_output)
    logger.info('Run name : {run_name}'.format(run_name=run_name))
    logger.info(config)

    logger.info('Constructing dictionaries...')
    dictionary = IndexDictionary.load(dictionary_path=dictionary_path,
                                      vocabulary_size=vocabulary_size)
    vocabulary_size = len(dictionary)
    #logger.info(f'dictionary vocabulary : {vocabulary_size} tokens')
    logger.info('dictionary vocabulary : {vocabulary_size} tokens'.format(vocabulary_size=vocabulary_size))

    logger.info('Loading datasets...')
    dataset = PairedDataset(data_path=data_path, dictionary=dictionary, dataset_limit=dataset_limit)
    logger.info('dataset size : {dataset_size}'.format(dataset_size=len(dataset)))

    logger.info('Building model...')
    model = build_model(layers_count, hidden_size, heads_count, d_ff, dropout_prob, max_len, vocabulary_size)
    model = stateLoading(model, checkpoint)
    model = disable_grad(model)
    model.to(device=device)

    logger.info(model)
    logger.info('{parameters_count} parameters'.format(
        parameters_count=sum([p.nelement() for p in model.parameters()])))

    if torch.cuda.device_count() > 1:
        loss_model = DataParallel(loss_model, output_device=1)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pretraining_collate_function)

    true_sequences = []
    predicted_sequences = []
    hints = []
    confidences = []
    for inputs, targets, batch_count in tqdm(dataloader, ncols=60):
        inputs = convert_to_tensor(inputs, device)
        targets = convert_to_tensor(targets, device)

        token_predictions, _ = model(inputs)
        token_predictions = F.softmax(token_predictions, dim=-1)
        token_targets = targets[0]
        indexed_sequences, _ = inputs
        for t, p, g in zip(indexed_sequences, token_predictions, token_targets):
            tmp_input = [dictionary.index_to_token(i.item()) for i in t]
            tmp_pred = [(dictionary.index_to_token(torch.argmax(i).item()), torch.max(i).item()) for i in p]
            tmp_target = [dictionary.index_to_token(i.item()) for i in g]
            tmp_input = tmp_input[1:]
            tmp_pseudo = tmp_input.copy()
            tmp_pred = tmp_pred[1:]
            tmp_target = tmp_target[1:]
            tmp_hint = ['='] * len(tmp_input)
            prob_num = 0.
            prob_denom = 0
            while MASK_TOKEN in tmp_input:
                index = tmp_input.index(MASK_TOKEN)
                tmp_input[index] = tmp_target[index]
                p = tmp_pred[index]
                tmp_pseudo[index] = p[0]
                prob_num += p[1]
                prob_denom += 1
                tmp_hint[index] = '*'
            if prob_denom == 0:
                continue
            prob = prob_num / prob_denom

            if prob > confidence:
                true_sequences.append(' '.join(tmp_input).replace(PAD_TOKEN, ''))
                predicted_sequences.append(' '.join(tmp_pseudo).replace(PAD_TOKEN, ''))
                hints.append(' '.join(tmp_hint))
                confidences.append(prob)
            # print(' '.join(tmp_input).replace(PAD_TOKEN, ''))
            # print(' '.join(tmp_pseudo).replace(PAD_TOKEN, ''))

    with open(pseudo_data_path, 'w') as f:
        for t, p, h, c in zip(true_sequences, predicted_sequences, hints, confidences):
            f.write('%s\n' % t)
            f.write('%s\n' % p)
            f.write('%s\n' % ''.join(h))
            f.write('confidence: %s\n' % c)
            f.write('-\n')
        f.close()
