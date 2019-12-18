import torch 
from torch import nn
import torch.nn.functional as F

from bert.train.senet.models import SELayer, SEBasicBlock

from bert.train.utils.onehot import index2onehot

class AMPscanner(nn.Module):

    def __init__(self, model, embedding_vector_length, nbf, flen, nlstm, ndrop):
        super(AMPscanner, self).__init__()
        self.model = model
        self.kernel_size = flen
        self.embedding = nn.Embedding(29, embedding_vector_length)
        self.conv = nn.Conv1d(embedding_vector_length, nbf, kernel_size=flen, stride=1, padding=(self.kernel_size // 2))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(5)
        self.lstm = nn.LSTM(nbf, nlstm, batch_first=True, bias=True, dropout=ndrop)
        self.linear = nn.Linear(nlstm, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, targets):
        
        # padded_sequences, padded_segments = inputs
        # out = self.embedding(padded_sequences)
        padded_sequences, padded_segments = inputs
        onehot = index2onehot(29, padded_sequences, padded_sequences.device)

        _, _, out = self.model(inputs)
        # out = torch.cat([out, onehot], dim=2)

        out = out.transpose(2, 1)
        out = self.conv(out)
        out = self.relu(out)
        out = self.pool(out)
        out = out.transpose(2, 1)
        out, (_, _) = self.lstm(out)
        out = out[:, -1, :]
        out = self.linear(out)
        out = self.sigmoid(out)
        out = torch.squeeze(out)
        targets = targets.to(torch.float32)
        loss = F.binary_cross_entropy(out, targets)
        prediction = (out >= 0.5).to(torch.int64)
        return prediction, loss.unsqueeze(dim=0)

class AMPscanner_SEnet(nn.Module):

    def __init__(self, model, embedding_vector_length, nbf, flen, nlstm, ndrop):
        super(AMPscanner_SEnet, self).__init__()
        self.model = model
        self.kernel_size = flen
        self.embedding = nn.Embedding(29, embedding_vector_length)
        self.conv1 = nn.Conv1d(embedding_vector_length, embedding_vector_length, kernel_size=15, stride=1, padding=(self.kernel_size // 2))
        self.relu1 = nn.ReLU()
        self.se1 = SELayer(embedding_vector_length)
        self.dropout1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(embedding_vector_length, embedding_vector_length, kernel_size=7, stride=1, padding=(self.kernel_size // 2))
        self.relu2 = nn.ReLU()
        self.se2 = SELayer(embedding_vector_length)
        self.dropout2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv1d(embedding_vector_length, nbf, kernel_size=3, stride=1, padding=(self.kernel_size // 2))
        self.relu3 = nn.ReLU()
        self.se3 = SELayer(nbf)
        # downsample = nn.Conv1d(embedding_vector_length, nbf, kernel_size=3, stride=1,\
        #                          padding=(3 // 2), bias=False)
        # self.se1 = SEBasicBlock(nbf, nbf, 3, downsample=None, reduction=2)
        # self.se2 = SEBasicBlock(nbf, nbf, 3, downsample=None, reduction=2)
        self.pool = nn.MaxPool1d(5)
        self.lstm = nn.LSTM(nbf, nlstm, batch_first=True, bias=True, dropout=ndrop)
        self.linear = nn.Linear(nlstm, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, targets):
        
        # padded_sequences, padded_segments = inputs
        # out = self.embedding(padded_sequences)
        _, _, out = self.model(inputs)
        out = out.transpose(2, 1)

        out = self.conv1(out)
        out = self.relu1(out)
        out = self.se1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.se2(out)
        out = self.dropout2(out)

        out = self.conv3(out)
        out = self.relu3(out)
        out = self.se3(out)
        out = F.layer_norm(out, out.size())

        # out = self.se1(out)
        # out = F.layer_norm(out, out.size())
        # out = self.conv(out)
        # out = self.relu(out)
        # out = F.layer_norm(out, out.size())
        # out = self.se2(out)
        # out = F.layer_norm(out, out.size())
        # out = self.se2(out)
        # out = F.layer_norm(out, out.size())
        out = self.pool(out)
        out = out.transpose(2, 1)
        out, (_, _) = self.lstm(out)
        # out = F.layer_norm(out, out.size())
        out = out[:, -1, :]
        out = self.linear(out)
        out = self.sigmoid(out)
        out = torch.squeeze(out)
        targets = targets.to(torch.float32)
        loss = F.binary_cross_entropy(out, targets)
        prediction = (out >= 0.5).to(torch.int64)
        return prediction, loss.unsqueeze(dim=0)

class AMPscanner_SEnet_poyu(nn.Module):

    def __init__(self, model, embedding_vector_length, nbf, flen, nlstm, ndrop):
        super(AMPscanner_SEnet_poyu, self).__init__()
        self.model = model
        self.kernel_size = flen
        self.embedding = nn.Embedding(29, embedding_vector_length)

        filters = 64
        reduction = 4
        self.conv = nn.Conv1d(embedding_vector_length, filters, kernel_size=3, stride=1, padding=(3//2))
        self.global_pool_1 = nn.AdaptiveAvgPool1d(1)
        self.downsample = nn.Linear(filters, int(filters/reduction))
        self.upsample = nn.Linear(int(filters/reduction), filters)
        self.pool = nn.AvgPool1d(5)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(nbf, nlstm, batch_first=True, bias=True, dropout=ndrop)
        self.global_pool_2 = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(2560, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, inputs, targets):
        
        # padded_sequences, padded_segments = inputs
        # out = self.embedding(padded_sequences)
        _, _, out = self.model(inputs)
        out = out.transpose(2, 1)

        out = self.conv(out)
        b, c, _ = out.size()
        res = out
        res = self.relu(res)
        res = self.global_pool_1(res)
        res = res.squeeze()

        res = self.downsample(res)
        res = self.relu(res)
        res = self.upsample(res)
        res = self.sigmoid(res)
        res = res.view(b, c, 1)
        out = out * res.expand_as(out)

        out = self.pool(out)
        out = self.dropout(out)
        out = out.transpose(2, 1)
        # out, (_, _) = self.lstm(out)

        # out = self.global_pool_2(out)
        out = out.flatten(start_dim=1, end_dim=2)

        out = self.linear(out)
        out = self.sigmoid(out)
        out = torch.squeeze(out)
        targets = targets.to(torch.float32)
        loss = F.binary_cross_entropy(out, targets)
        prediction = (out >= 0.5).to(torch.int64)
        return prediction, loss.unsqueeze(dim=0)

class AMPscanner_BERT(nn.Module):

    def __init__(self, model, hidden_size, embedding_vector_length):
        super(AMPscanner_BERT, self).__init__()
        self.model = model

        self.conv1 = nn.Conv1d(200, 100, 15)
        self.max1 = nn.MaxPool1d(5)
        # self.conv2 = nn.Conv1d(100, 50, 13)
        # self.max2 = nn.MaxPool1d(5)
        self.linear = nn.Linear(3700, 1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, inputs, targets):
        
        # padded_sequences, padded_segments = inputs
        # out = self.embedding(padded_sequences)
        _, _, out = self.model(inputs)

        
        out = out.transpose(1, 2)
        out = self.conv1(out)
        out = self.max1(out)
        # out = self.conv2(out)
        # out = self.max2(out)
        out = out.transpose(1, 2)
    
        # out = self.linear_1(out)
        out = out.flatten(start_dim=1, end_dim=2)
        # out = self.linear_2(out)
        out = self.linear(out)
        out = self.sigmoid(out)
        out = torch.squeeze(out)
        targets = targets.to(torch.float32)
        loss = F.binary_cross_entropy(out, targets)
        prediction = (out >= 0.5).to(torch.int64)
        return prediction, loss.unsqueeze(dim=0)
