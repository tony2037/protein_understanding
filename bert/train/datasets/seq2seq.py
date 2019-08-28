from bert.preprocess import PAD_INDEX

class Seq2SeqDataset:

    def __init__(self, data_path, dictionary, fixed_length=None):

        self.data = []
        self.dimension = None
        self.fixed_length = fixed_length
        with open(data_path) as file:
            assert file.readline() == 'sentence\tlabel\n'
            self.dimension = int(file.readline())

            for line in file:
                tokenized_sentence, answer = line.strip().split('|')
                indexed_sentence = [dictionary.token_to_index(token) for token in tokenized_sentence.split()]
                label = [self.onehot(int(l)) for l in answer.split()]

                if self.fixed_length != None:
                    padding_length = self.fixed_length - len(indexed_sentence)
                    assert padding_length >= 0
                    padding_sentence = [PAD_INDEX for _ in range(padding_length)]
                    padding_label = [[0 for _ in range(self.dimension)] for _ in range(padding_length)]
                    indexed_sentence = indexed_sentence + padding_sentence
                    label = label + padding_label
                    assert len(indexed_sentence) == len(label) == self.fixed_length
                assert len(indexed_sentence) == len(label)
                self.data.append((indexed_sentence, label))

    def __getitem__(self, item):

        indexed_sentence, label = self.data[item]
        segment = [0] * len(indexed_sentence)
        return (indexed_sentence, segment), label

    def __len__(self):
        return len(self.data)

    def onehot(self, index):

        tmp = [0. for _ in range(self.dimension)]
        tmp[index] = 1.
        return tmp
