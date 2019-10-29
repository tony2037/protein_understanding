from bert.preprocess import PAD_INDEX

from torch import nn


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

"""
class CrossEntropyLossMutation(nn.Module):
    '''
        This criterion (`CrossEntropyLoss`) combines `LogSoftMax` and `NLLLoss` in one single class.
 
        NOTE: Computes per-element losses for a mini-batch (instead of the average loss over the entire mini-batch).
    '''
    def __init__(self, mutation_matrix):

        super().__init__()
        self.log_softmax = nn.LogSoftmax()
        self.mutation_matrix = autograd.Variable(mutation_matrix)

        def forward(self, logits, target):

            log_probabilities = self.log_softmax(logits)
            # NLLLoss(x, class) = -weights[class] * x[class]
            return -self.class_weights.index_select(0, target) * log_probabilities.index_select(-1, target).diag()
"""

class MutaionCrossEntropyLoss(nn.Module):

    def __init__(self, mutation_matrix, ignore_index):

        super(MutationCrossEntropyLoss, self).__init__()
        self.mutation_matrix = mutation_matrix
        self.ignore_index = ignore_index

    def forward(self, input, target):

        loss = torch.tensor(0., requires_grad=True, dtype=torch.float32)
        n = input.size(0)
        c = input.size(1)

        for i in range(n):
            _input = input[i].unsqueeze(0)
            _target = target[i].unsqueeze(0)
            matrix_index = int(_target.itemt())
            if matrix_index == self.ignore_index:
                continue
            weight = self.mutation_matrix[matrix_index]
            loss = loss.add(F.cross_entropy(_input, _target, weight=weight, ignore_index=self.ignore_index))
        loss = loss.divide(n)
        return loss

class MutationMLMLossModel(nn.Module):

    def __init__(self, model, mutation_matrix):
        super(MLMLossModel, self).__init__()

        self.model = model
        self.mlm_loss_function = MutaionCrossEntropyLoss(mutation_matrix=mutation_matrix, ignore_index=PAD_INDEX)
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
