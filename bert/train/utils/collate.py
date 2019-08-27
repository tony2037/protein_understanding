from bert.preprocess import PAD_INDEX


def pretraining_collate_function(batch):

    targets = [target for _, (target, is_next) in batch]
    longest_target = max(targets, key=lambda target: len(target))
    max_length = len(longest_target)

    padded_sequences = []
    padded_segments = []
    padded_targets = []
    is_nexts = []

    for (sequence, segment), (target, is_next) in batch:
        length = len(sequence)
        padding = [PAD_INDEX] * (max_length - length)
        padded_sequence = sequence + padding
        padded_segment = segment + padding
        padded_target = target + padding

        padded_sequences.append(padded_sequence)
        padded_segments.append(padded_segment)
        padded_targets.append(padded_target)
        is_nexts.append(is_next)

    count = 0
    for target in targets:
        for token in target:
            if token != PAD_INDEX:
                count += 1

    return (padded_sequences, padded_segments), (padded_targets, is_nexts), count


def classification_collate_function(batch):

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

        padded_sequences.append(padded_sequence)
        padded_segments.append(padded_segment)
        labels.append(label)

    count = len(labels)

    return (padded_sequences, padded_segments), labels, count

def seq2seq_collate_function(batch):

    """
    @ batch: (indexed_sentence, segment), label
    @@ indexed_sentence: A sery of a indexs that represent the tokens of a sentence (according to the given dictionary)
    @@ segment: [0] * len(indexed_sentence)
    @@ label: one-hot encoding, (len(indexed_sentence), dimension) i.e given dimension=2 [[0, 1], [0, 1], [1, 0] ...]

    Return: (padded_sequences, padded_segments), labels, count
    @ padded_sequences: indexed_sentences with padding, eg. [2, 3, 13, 9, ..., 0, 0, ...] (0 represents padding)
    @ padded_segments: segment with padding, eg. [0, 0, 0, ...], actually is literally all zeros and the length is same as\
                       padded_sequences, and labels (assume [PAD_INDEX] is 0)
    @ labels: One-hot encoding with padding, a padding'd be like [0, 0, 0 ...] depends on the dimesion of original label\
              , which is literally how many classes are in this task 
    @ count: the amount of samples this batch 
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

        label_padding = [[0 for _ in range(len(label[0]))] for _ in  range(max_length - length)]
        padded_label = label + label_padding

        padded_sequences.append(padded_sequence)
        padded_segments.append(padded_segment)
        labels.append(padded_label)

    count = len(labels)

    return (padded_sequences, padded_segments), labels, count
