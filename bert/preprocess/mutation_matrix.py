import torch
from torch.nn import functional as F

from . import PAD_TOKEN, UNK_TOKEN, MASK_TOKEN, CLS_TOKEN, SEP_TOKEN
import json

class MutationMatrix:

    def __init__(self, dictionary, matrix_path):

        self.dictionary = dictionary
        vocabulary_size = len(dictionary)
        self.matrix = [[1 for _ in  range(vocabulary_size)] for _ in range(vocabulary_size)]
        with open(matrix_path) as f:
            data = json.load(f)
            for token in data:
                index = dictionary.token_to_index(token)
                array = self.build_array(data[token])
                self.matrix[index] = array

        self.matrix = torch.tensor(self.matrix, dtype=torch.float32)

    def build_array(self, array):

        result = [1 for _ in range(len(self.dictionary))]
        for token in array:
            index = self.dictionary.token_to_index(token)
            if index != self.dictionary.token_to_index(UNK_TOKEN):
                value = array[token]
                result[index] = value

        return result

    def get_matrix(self):
        return self.matrix

    def get_softmax(self):
        tmp = self.matrix.detach().clone()
        prob_matrix = []
        for v in tmp:
            p = 10 ** (v / 10)
            p = torch.unsqueeze(p, dim=0)
            prob_matrix.append(p)
        prob_matrix = torch.cat(prob_matrix, dim=0)
        prob_matrix = F.softmax(prob_matrix, dim=1)
        return prob_matrix
