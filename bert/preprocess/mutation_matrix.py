from . import PAD_TOKEN, UNK_TOKEN, MASK_TOKEN, CLS_TOKEN, SEP_TOKEN
import json

class MutationMatrix:

    def __init__(self, dictionary, matrix_path):

        self.dictionary = dictionary
        vocabulary_size = len(dictionary)
        self.matrix = [0 for _ in range(vocabulary_size)]
        with open(matrix_path) as f:
            data = json.load(f)
            for token in data:
                index = dictionary.token_to_index(token)
                array = self.build_array(data[token])
                self.matrix[index] = array

        print(self.matrix)

    def build_array(self, array):

        result = [0 for _ in range(len(self.dictionary))]
        for token in array:
            index = self.dictionary.token_to_index(token)
            value = array[token]
            result[index] = value

        return result
