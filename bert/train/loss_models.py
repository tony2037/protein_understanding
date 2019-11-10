from bert.preprocess import PAD_INDEX
from bert.train import IGNORE_INDEX

import torch
from torch import nn
from torch.nn import functional as F


class MLMNSPLossModel(nn.Module):

    def __init__(self, model):
        super(MLMNSPLossModel, self).__init__()

        self.model = model
        self.mlm_loss_function = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)
        self.nsp_loss_function = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):

        outputs = self.model(inputs)

        mlm_outputs, nsp_outputs = outputs
        mlm_targets, is_nexts = targets

        mlm_predictions, nsp_predictions = mlm_outputs.argmax(dim=2), nsp_outputs.argmax(dim=1)
        predictions = (mlm_predictions, nsp_predictions)

        batch_size, seq_len, vocabulary_size = mlm_outputs.size()

        mlm_outputs_flat = mlm_outputs.view(batch_size * seq_len, vocabulary_size)
        mlm_targets_flat = mlm_targets.view(batch_size * seq_len)

        mlm_loss = self.mlm_loss_function(mlm_outputs_flat, mlm_targets_flat)
        nsp_loss = self.nsp_loss_function(nsp_outputs, is_nexts)

        loss = mlm_loss + nsp_loss

        return predictions, loss.unsqueeze(dim=0)

class MLMLossModel(nn.Module):

    def __init__(self, model):
        super(MLMLossModel, self).__init__()

        self.model = model
        self.mlm_loss_function = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)
        self.nsp_loss_function = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):

        outputs = self.model(inputs)

        mlm_outputs, nsp_outputs = outputs
        mlm_targets, is_nexts = targets

        mlm_predictions, nsp_predictions = mlm_outputs.argmax(dim=2), nsp_outputs.argmax(dim=1)
        predictions = (mlm_predictions, nsp_predictions)

        batch_size, seq_len, vocabulary_size = mlm_outputs.size()

        mlm_outputs_flat = mlm_outputs.view(batch_size * seq_len, vocabulary_size)
        mlm_targets_flat = mlm_targets.view(batch_size * seq_len)

        mlm_loss = self.mlm_loss_function(mlm_outputs_flat, mlm_targets_flat)
        nsp_loss = self.nsp_loss_function(nsp_outputs, is_nexts)

        loss = mlm_loss

        return predictions, loss.unsqueeze(dim=0)

class ClassificationLossModel(nn.Module):

    def __init__(self, model):
        super(ClassificationLossModel, self).__init__()

        self.model = model
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):

        outputs = self.model(inputs)
        predictions = outputs.argmax(dim=1)
        loss = self.loss_function(outputs, targets)

        return predictions, loss.unsqueeze(dim=0)

class PretrainSeq2SeqLossModel(nn.Module):

    def __init__(self, model, hidden_size, num_class):
        super(PretrainSeq2SeqLossModel, self).__init__()
        
        self.model = model
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.classification_layer = nn.Linear(hidden_size, num_class)
        self.classification_loss_function = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def forward(self, inputs, targets):
        
        _, _, encoded_sources = self.model(inputs)
        (batch, length, channel) = encoded_sources.size()
        out = torch.zeros((batch, length, self.num_class), device=encoded_sources.device)
        for b in range(batch):
            for l in range(length):
                tmp = encoded_sources[b][l].unsqueeze(0)
                tmp = self.classification_layer(tmp)
                tmp = tmp.squeeze()
                out[b][l] = tmp
        predictions = out.argmax(dim=2)
        flatten_out = out.flatten(start_dim=0, end_dim=1)
        flatten_targets = targets.flatten(start_dim=0, end_dim=1)
        loss = self.classification_loss_function(flatten_out, flatten_targets)

        return predictions, loss.unsqueeze(dim=0)
