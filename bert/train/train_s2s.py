from bert.preprocess.dictionary import IndexDictionary
from .model.bert import build_model
from .datasets.NoOneHot import Seq2SeqDataset
from .trainer import Trainer
from .utils.log import make_run_name, make_logger, make_checkpoint_dir
from .utils.stateload import stateLoading
from .optimizers import NoamOptimizer
from torch.optim import Adam

from .Seq2Seq.utils import Seq2Seq_Metric

from .sigunet.sigunet import sigunet
from .concate.concate import concate

import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader

import random
import numpy as np
from os.path import join

RUN_NAME_FORMAT = (
    "BERT-"
    "{phase}-"
    "layers_count={layers_count}-"
    "hidden_size={hidden_size}-"
    "heads_count={heads_count}-"
    "{timestamp}"
)

def finetuneSeq2Seq(pretrained_checkpoint,
             data_dir, train_path, val_path, dictionary_path,
             vocabulary_size, batch_size, max_len, epochs, lr, clip_grads, device,
             layers_count, hidden_size, heads_count, d_ff, dropout_prob,
             log_output, checkpoint_dir, print_every, save_every, config, run_name=None, fixed_length=None, **_):

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    train_path = train_path if data_dir is None else join(data_dir, train_path)
    val_path = val_path if data_dir is None else join(data_dir, val_path)
    dictionary_path = dictionary_path if data_dir is None else join(data_dir, dictionary_path)

    run_name = run_name if run_name is not None else make_run_name(RUN_NAME_FORMAT, phase='finetune', config=config)
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
    train_dataset = Seq2SeqDataset(data_path=train_path, dictionary=dictionary, fixed_length=fixed_length)
    val_dataset = Seq2SeqDataset(data_path=val_path, dictionary=dictionary, fixed_length=fixed_length)
    logger.info('Train dataset size : {dataset_size}'.format(dataset_size=len(train_dataset)))

    logger.info('Building model...')
    pretrained_model = build_model(layers_count, hidden_size, heads_count, d_ff, dropout_prob, max_len, vocabulary_size, forward_encoded=True)
    pretrained_model = stateLoading(pretrained_model, pretrained_checkpoint)

    concate_model = concate(output_size=64, hidden_size=hidden_size, onehot_size=vocabulary_size)
    model = sigunet(model=pretrained_model, concate=concate_model, m=28, n=4, kernel_size=7, pool_size=2, threshold=0.1, device=device)

    logger.info(model)
    logger.info('{parameters_count} parameters'.format(
        parameters_count=sum([p.nelement() for p in model.parameters()])))

    # Have not figured this out yet
    metric_functions = []

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=Seq2SeqDataset.collate_function)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=Seq2SeqDataset.collate_function)

    optimizer = Adam(model.parameters(), lr=lr)

    checkpoint_dir = make_checkpoint_dir(checkpoint_dir, run_name, config)

    logger.info('Start training...')
    trainer = Trainer(
        loss_model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        metric_functions=metric_functions,
        optimizer=optimizer,
        clip_grads=clip_grads,
        logger=logger,
        checkpoint_dir=checkpoint_dir,
        print_every=print_every,
        save_every=save_every,
        device=device
    )

    trainer.run(epochs=epochs)
    return trainer
