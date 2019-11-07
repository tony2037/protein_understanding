from bert.preprocess.dictionary import IndexDictionary
from .model.bert import build_model, FineTuneModel
from .datasets.NoOneHot import Seq2SeqDataset
from .trainer import Trainer
from .utils.log import make_run_name, make_logger, make_checkpoint_dir
from .utils.fix_weights import disable_grad
from .utils.stateload import stateLoading, remove_state
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .sigunet.sigunet import sigunet
from .Seq2Seq.utils import Seq2Seq_Metric

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

def finetuneSigunet(pretrained_checkpoint,
             data_dir, train_path, val_path, dictionary_path,
             vocabulary_size, batch_size, max_len, epochs, lr, clip_grads, device,
             layers_count, hidden_size, heads_count, d_ff, dropout_prob,
             log_output, checkpoint_dir, print_every, save_every, config, run_name=None, **_):

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
    train_dataset = Seq2SeqDataset(data_path=train_path, dictionary=dictionary)
    val_dataset = Seq2SeqDataset(data_path=val_path, dictionary=dictionary)
    logger.info('Train dataset size : {dataset_size}'.format(dataset_size=len(train_dataset)))

    logger.info('Building model...')
    pretrained_model = build_model(layers_count, hidden_size, heads_count, d_ff, dropout_prob, max_len, vocabulary_size, forward_encoded=True)
    to_removes = ['classification_layer.weight', 'classification_layer.bias']
    state_dict = torch.load(pretrained_checkpoint, map_location='cpu')['state_dict']
    state_dict = remove_state(state_dict, to_removes)
    pretrained_model.load_state_dict(state_dict)
    pretrained_model = disable_grad(pretrained_model)
    #pretrained_model.eval()

    model = sigunet(model=pretrained_model, m=28, n=4, kernel_size=7, pool_size=2, threshold=0.1, device=device)

    logger.info(model)
    logger.info('{parameters_count} parameters'.format(
        parameters_count=sum([p.nelement() for p in model.parameters()])))

    # Have not figured this out yet
    metric_functions = [Seq2Seq_Metric]

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=Seq2SeqDataset.collate_function)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=Seq2SeqDataset.collate_function)

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
        monitor='train_loss',
        comment='sigunet_new-pretrain'
    )

    trainer.run(epochs=epochs)
    return trainer
