from torch import nn
from bert.preprocess import PAD_INDEX
import numpy as np

def build_pcnn(vocabulary_size, hidden_size, in_channels, out_channels, kernel_sizes, acts):

    token_embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=hidden_size)

    return P_CNN(vocabulary_size, hidden_size, token_embedding, in_channels, out_channels, kernel_sizes, acts)

def mlm_accuracy(predictions, targets):
    mlm_predictions = predictions
    mlm_targets = targets

    relevent_indexes = np.where(mlm_targets != PAD_INDEX)
    relevent_predictions = mlm_predictions[relevent_indexes]
    relevent_targets = mlm_targets[relevent_indexes]

    corrects = np.equal(relevent_predictions, relevent_targets)
    return corrects.mean()

class MLMLossModel(nn.Module):

    def __init__(self, model):
        super(MLMLossModel, self).__init__()

        self.model = model
        self.mlm_loss_function = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)

    def forward(self, inputs, targets):

        outputs = self.model(inputs)

        mlm_outputs, _ = outputs
        mlm_targets = targets

        predictions = mlm_predictions = mlm_outputs.argmax(dim=2)

        batch_size, seq_len, vocabulary_size = mlm_outputs.size()

        mlm_outputs_flat = mlm_outputs.view(batch_size * seq_len, vocabulary_size)
        mlm_targets_flat = mlm_targets.view(batch_size * seq_len)

        mlm_loss = self.mlm_loss_function(mlm_outputs_flat, mlm_targets_flat)

        loss = mlm_loss

        return predictions, loss.unsqueeze(dim=0)

class P_CNN(nn.Module):

    def __init__(self, vocabulary_size, hidden_size, token_embedding, in_channels, out_channels, kernel_sizes, acts):
        super(P_CNN, self).__init__()
        """
        Protein CNN

        args:
            @ hidden_size (int): The hidden size shall be the output hidden size, which is, out_channels[-1]
                                , and the self.token_prediction_layer would predict tokens at every position.
            @ token_embedding (nn.Module): The model of encoding, no matter one-hot or embedding, you can specific it.

        """

        self.token_embedding = token_embedding
        self.token_prediction_layer = nn.Linear(hidden_size, vocabulary_size)
        self.cnn = CNN(in_channels, out_channels, kernel_sizes, acts)

    def forward(self, inputs):
        
        sequences, segments = inputs
        token_embedded = self.token_embedding(sequences)
        encoded_sources = self.cnn(token_embedded)

        token_predictions = self.token_prediction_layer(encoded_sources)

        return token_predictions, encoded_sources

        
class CNN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes, acts):
        super(CNN, self).__init__()
        
        assert len(in_channels) == len(out_channels) == len(kernel_sizes) == len(acts)
        self.cnn_layers = nn.ModuleList(
            [CNNLayer(i_c, o_c, k, a) for i_c, o_c, k, a in zip(in_channels, out_channels, kernel_sizes, acts)]        
        )

    def forward(self, x):
        """

        args:
            x: inputs, (batch_size, seq_len, embed_size)
        """
        out = x.transpose(1, 2)
        for cnn_layer in self.cnn_layers:
            out = cnn_layer(out)
        out = out.transpose(1, 2)

        return out

class CNNLayer(nn.Module):

    def __init__(self, in_channels, out_channels=None, kernel_size=None, act=nn.ReLU()):
        super(CNNLayer, self).__init__()
        """1d convoluation operation.

        Arguments:
            input (torch.Tensor): Input tensor to do convoluation.
                                                Its shape need to be `(batch_size, channels, length)`.
            in_channels (int): Number of the input channels.
            out_channels (int): Number of the output channels.
            kernel_size (int): Length of 1d kernel.
            act (activation function): Activation function after convoluation. Default is `nn.ReLU`.

        Returns:
            Tensor
        """

        assert (kernel_size & 1) == 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Since there is no padding='SAME' such feature, refer to https://discuss.pytorch.org/t/convolution-1d-and-simple-function/11606/5
        # which says that by setting padding=(kernel_size // 2) can nail same objective provided kernel size is odd
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(self.kernel_size // 2))
        self.act = act

    def forward(self, input):

        out = self.conv(input)
        out = self.act(out)
        return out
