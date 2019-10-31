from bert.preprocess import PAD_INDEX

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

class MutationCrossEntropyLoss(nn.Module):

    def __init__(self, mutation_matrix, ignore_index, device):

        super(MutationCrossEntropyLoss, self).__init__()
        self.mutation_matrix = mutation_matrix
        self.ignore_index = ignore_index
        self.device = device if device is not None else 'cpu'

    def forward(self, input, target):

        loss = self.customized_cross_entropy(input, target)
        return loss

    def customized_cross_entropy(self, input, target):
        """
        input: (N, C)
        target: (N)
        """
        loss = torch.tensor(0., requires_grad=True, dtype=torch.float32, device=self.device)
        n = input.size(0)
        c = input.size(1)
        assert c == self.mutation_matrix.size(0)

        for i in range(n):
            _input = F.softmax(input[i], dim=0)
            _target = target[i]
            matrix_index = int(_target.item())
            if matrix_index == self.ignore_index:
                continue
            weight = self.mutation_matrix[matrix_index]
            for j in range(c):
                loss = loss.add(-weight[j] * torch.log(_input[j]))

        loss = loss.div(n)
        return loss

class MutationMLMLossModel(nn.Module):

    def __init__(self, model, mutation_matrix, device):
        super(MutationMLMLossModel, self).__init__()

        self.device = device if device is not None else 'cpu'
        self.model = model
        self.mutation_matrix = mutation_matrix.to(device=self.device)
        self.mlm_loss_function = MutationCrossEntropyLoss(mutation_matrix=self.mutation_matrix, ignore_index=PAD_INDEX, device=self.device)
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
