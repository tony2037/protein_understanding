from bert.preprocess.dictionary import IndexDictionary
from .model.bert import build_model
from .datasets.NoOneHot import ClassificationDataset
from .trainer import Trainer
from .utils.log import make_run_name, make_logger, make_checkpoint_dir
from .utils.stateload import stateLoading
from .utils.fix_weights import disable_grad
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .AMP.model import AMPscanner, AMPscanner_SEnet, AMPscanner_SEnet_poyu
from .AMP.utils import evaluator

import torch
from torch import nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader

import random
import numpy as np
from os.path import join

RUN_NAME_FORMAT = (
    "AMPscanner-"
    "{phase}-"
    "layers_count={layers_count}-"
    "hidden_size={hidden_size}-"
    "heads_count={heads_count}-"
    "{timestamp}"
)

def finetuneAMPscanner(pretrained_checkpoint,
             data_dir, train_path, val_path, dictionary_path,
             vocabulary_size, batch_size, max_len, epochs, lr, clip_grads, device,
             layers_count, hidden_size, heads_count, d_ff, dropout_prob,
             log_output, checkpoint_dir, print_every, save_every, embedding_vector_length, nbf, flen, nlstm, ndrop,
             config, run_name=None, fixed_length=None, **_):

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
    train_dataset = ClassificationDataset(data_path=train_path, dictionary=dictionary)
    val_dataset = ClassificationDataset(data_path=val_path, dictionary=dictionary)
    logger.info('Train dataset size : {dataset_size}'.format(dataset_size=len(train_dataset)))

    logger.info('Building model...')
    pretrained_model = build_model(layers_count, hidden_size, heads_count, d_ff, dropout_prob, max_len, vocabulary_size, forward_encoded=True)
    pretrained_model = stateLoading(pretrained_model, pretrained_checkpoint)
    pretrained_model = disable_grad(pretrained_model)
    pretrained_model.eval()

    model = AMPscanner(model=pretrained_model, embedding_vector_length=embedding_vector_length, nbf=nbf, flen=flen, nlstm=nlstm, ndrop=ndrop)

    logger.info(model)
    logger.info('{parameters_count} parameters'.format(
        parameters_count=sum([p.nelement() for p in model.parameters()])))

    # Have not figured this out yet
    eva = evaluator()
    metric_functions = [eva.MCC]

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ClassificationDataset.collate_function)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ClassificationDataset.collate_function)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=3)

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
        device=device,
        scheduler=scheduler,
        monitor='val_loss',
        comment = 'AMPscanner_Reproduce'
    )

    trainer.run(epochs=epochs)
    return trainer

def finetuneAMPscanner_SEnet(pretrained_checkpoint,
             data_dir, train_path, val_path, dictionary_path,
             vocabulary_size, batch_size, max_len, epochs, lr, clip_grads, device,
             layers_count, hidden_size, heads_count, d_ff, dropout_prob,
             log_output, checkpoint_dir, print_every, save_every, embedding_vector_length, nbf, flen, nlstm, ndrop,
             config, run_name=None, fixed_length=None, **_):

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
    train_dataset = ClassificationDataset(data_path=train_path, dictionary=dictionary)
    val_dataset = ClassificationDataset(data_path=val_path, dictionary=dictionary)
    logger.info('Train dataset size : {dataset_size}'.format(dataset_size=len(train_dataset)))

    logger.info('Building model...')
    pretrained_model = build_model(layers_count, hidden_size, heads_count, d_ff, dropout_prob, max_len, vocabulary_size, forward_encoded=True)
    pretrained_model = stateLoading(pretrained_model, pretrained_checkpoint)
    pretrained_model = disable_grad(pretrained_model)
    pretrained_model.eval()

    model = AMPscanner_SEnet_poyu(model=pretrained_model, embedding_vector_length=embedding_vector_length, nbf=nbf, flen=flen, nlstm=nlstm, ndrop=ndrop)

    logger.info(model)
    logger.info('{parameters_count} parameters'.format(
        parameters_count=sum([p.nelement() for p in model.parameters()])))

    # Have not figured this out yet
    eva = evaluator()
    metric_functions = [eva.MCC]

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ClassificationDataset.collate_function)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ClassificationDataset.collate_function)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=3)

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
        device=device,
        scheduler=scheduler,
        monitor='val_loss',
        comment = 'AMPscanner_Reproduce'
    )

    trainer.run(epochs=epochs)
    return trainer

def finetuneAMPscanner_BERT(pretrained_checkpoint,
             data_dir, train_path, val_path, dictionary_path,
             vocabulary_size, batch_size, max_len, epochs, lr, clip_grads, device,
             layers_count, hidden_size, heads_count, d_ff, dropout_prob,
             log_output, checkpoint_dir, print_every, save_every, embedding_vector_length, nbf, flen, nlstm, ndrop,
             config, run_name=None, fixed_length=None, **_):

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
    train_dataset = ClassificationDataset(data_path=train_path, dictionary=dictionary)
    val_dataset = ClassificationDataset(data_path=val_path, dictionary=dictionary)
    logger.info('Train dataset size : {dataset_size}'.format(dataset_size=len(train_dataset)))

    logger.info('Building model...')
    # from bert.train.model.cnn import build_cbert
    # in_channels = [200, 100, 50, 100]
    # out_channels = [100, 50, 100, 200]
    # kernel_sizes = [7, 5, 3, 1]
    # acts = [nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()]
    # pretrained_model = build_cbert(hidden_size, vocabulary_size, in_channels, out_channels, kernel_sizes, acts,\
    #             layers_count, heads_count, d_ff, dropout_prob, max_len)
    pretrained_model = build_model(layers_count, hidden_size, heads_count, d_ff, dropout_prob, max_len, vocabulary_size, forward_encoded=True)

    from .AMP.model import AMPscanner_BERT
    model = AMPscanner_BERT(pretrained_model, hidden_size, embedding_vector_length)

    logger.info(model)
    logger.info('{parameters_count} parameters'.format(
        parameters_count=sum([p.nelement() for p in model.parameters()])))

    # Have not figured this out yet
    eva = evaluator()
    metric_functions = [eva.MCC]

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ClassificationDataset.collate_function)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ClassificationDataset.collate_function)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=3)

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
        device=device,
        scheduler=scheduler,
        monitor='val_loss',
        comment = 'AMPscanner_Reproduce'
    )

    trainer.run(epochs=epochs)
    return trainer
