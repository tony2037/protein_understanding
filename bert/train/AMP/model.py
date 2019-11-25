import torch 
from torch import nn
import torch.nn.functional as F

from bert.train.senet.models import SEBasicBlock

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
        _, _, out = self.model(inputs)
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
        downsample = nn.Conv1d(embedding_vector_length, nbf, kernel_size=3, stride=1,\
                                 padding=(3 // 2), bias=False)
        self.se1 = SEBasicBlock(embedding_vector_length, nbf, flen, downsample=downsample, reduction=16)
        self.se2 = SEBasicBlock(nbf, nbf, flen, downsample=None, reduction=16)
        self.pool = nn.MaxPool1d(5)
        self.lstm = nn.LSTM(nbf, nlstm, batch_first=True, bias=True, dropout=ndrop)
        self.linear = nn.Linear(nlstm, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, targets):
        
        # padded_sequences, padded_segments = inputs
        # out = self.embedding(padded_sequences)
        _, _, out = self.model(inputs)
        out = out.transpose(2, 1)
        out = self.se1(out)
        out = self.se2(out)
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
