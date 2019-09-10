from bert.preprocess import PAD_INDEX
from bert.train import IGNORE_INDEX

class Seq2SeqDataset:

    def __init__(self, data_path, dictionary, fixed_length=None):

        self.data = []
        self.dimension = None
        self.fixed_length = fixed_length
        with open(data_path) as file:
            assert file.readline() == 'sentence\tlabel\n'

            for line in file:
                tokenized_sentence, answer = line.strip().split('|')
                indexed_sentence = [dictionary.token_to_index(token) for token in tokenized_sentence.split()]
                label = [int(l) for l in answer.split()]
                assert len(indexed_sentence) == len(label)
                self.data.append((indexed_sentence, label))

    @staticmethod
    def collate_function(batch):
        """
        This function is for collation and is an argument for torch.utils.data.DataLoader
        @ batch: (indexed_sentence, segment), label
        """
        lengths = [len(sequence) for (sequence, _), _ in batch]
        max_length = max(lengths)

        padded_sequences = []
        padded_segments = []
        labels = []

        for (sequence, segment), label in batch:
            length = len(sequence)
            padding = [PAD_INDEX] * (max_length - length)
            padded_sequence = sequence + padding
            padded_segment = segment + padding

            label_padding = [IGNORE_INDEX] * (max_length - length)
            padded_label = label + label_padding

            padded_sequences.append(padded_sequence)
            padded_segments.append(padded_segment)
            labels.append(padded_label)

        count = len(labels)

        return (padded_sequences, padded_segments), labels, count

    def __getitem__(self, item):

        indexed_sentence, label = self.data[item]
        segment = [0] * len(indexed_sentence)
        return (indexed_sentence, segment), label

    def __len__(self):
        return len(self.data)

class ClassificationDataset:

    def __init__(self, data_path, dictionary):

        self.data = []
        with open(data_path) as file:
            assert file.readline() == 'sentence\tlabel\n'

            for line in file:
                tokenized_sentence, label = line.strip().split('\t')
                indexed_sentence = [dictionary.token_to_index(token) for token in tokenized_sentence.split()]
                self.data.append((indexed_sentence, int(label)))

    def __getitem__(self, item):
        indexed_text, label = self.data[item]
        segment = [0] * len(indexed_text)
        return (indexed_text, segment), label

    def __len__(self):
        return len(self.data)
