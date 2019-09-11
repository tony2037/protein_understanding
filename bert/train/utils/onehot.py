import torch

def index2onehot(dim, indexed_sequences):
    """
    Transform indexed sequences to one-hot encodding vectores
    @ dim: The number of class, at most case is the length of dictionary or the number of categories of tokens
    @ indexed_sequences: (batch_size, length)

    Return: (batch_size, length, dim)
    """
    onehot = torch.zeros(indexed_sequences.shape[0], indexed_sequences.shape[1], dim)
    indexed_sequences = indexed_sequences.unsqueeze(2)
    return onehot.scatter_(2, indexed_sequences.to(torch.long), 1)
