import torch 
from torch import nn
import torch.nn.functional as F

class AMPscanner(nn.Module):

    def __init__(self, model, embedding_vector_length, nbf, flen, nlstm, ndrop):
        super(AMPscanner, self).__init__()
        self.model = model
        self.kernel_size = flen
        self.embedding = nn.Embedding(21, embedding_vector_length)
        self.conv = nn.Conv1d(embedding_vector_length, nbf, kernel_size=flen, stride=1, padding=(self.kernel_size // 2))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(5)
        self.lstm = nn.LSTM(nbf, nlstm, batch_first=True, bias=True, dropout=ndrop)
        self.linear = nn.Linear(nlstm, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out = self.embedding(input)
        out = out.transpose(2, 1)
        out = self.conv(out)
        out = self.relu(out)
        out = self.pool(out)
        out = out.transpose(2, 1)
        out, (_, _) = self.lstm(out)
        out = out[:, -1, :]
        out = self.linear(out)
        out = self.sigmoid(out)
        loss = F.binary_cross_entropy(out, y)
        prediction = (out >= 0.5).float()
        return prediction, loss.unsqueeze(dim=0)
