import numpy as np


def sentence_to_embeddings(sentence, target_sequence, pw, polarity, embeddings):
    sentence_to_embeddings = [
        embeddings[word] for word in sentence
        ]
    target_sequence_to_embeddings = [
        embeddings[target] for target in target_sequence
    ]

    return (
        sentence_to_embeddings, 
        target_sequence_to_embeddings,
        pw,
        polarity
        )

def get_sequence_length(batch):
    sentence_length = [
        len(item[0]) for item in batch
    ]

    target_legnth = [
        len(item[1]) for item in batch
    ]

    return sentence_length, target_legnth

def get_max_sequence_length(batch):
    sentence_length = [
        len(item[0]) for item in batch
    ]

    target_length = [
        len(item[1]) for item in batch
    ]

    max_sentence_length = max(sentence_length)
    max_target_length = max(target_length)

    sentence_length = [
        max_sentence_length for _ in sentence_length
    ]

    target_length = [
        max_target_length for _ in target_length
    ]

    return sentence_length, target_length

def batch_embeddings_padding(transformed_batch, embeddings):
    """
    Perform zero padding on transformed batch.
    """
    max_sentence_length = max(
        [len(item[0]) for item in transformed_batch]
    )

    max_target_length = max(
        [len(item[1]) for item in transformed_batch]
    )

    padded_sentence_batch, padded_target_batch, padded_pw_batch, labels = [], [], [], []

    for item in transformed_batch:
        padded_sentence = np.pad(
            item[0], 
            pad_width=((0, max_sentence_length - len(item[0])), (0, 0)),
            mode='constant'
            )
        # padded_sentence = item[0]

        # for _ in range(max_sentence_length - len(item[0])):
        #     padded_sentence = np.append(
        #         padded_sentence, 
        #         np.expand_dims(embeddings[''], axis=0),
        #         axis=0
        #     )

        padded_target = np.pad(
            item[1],
            pad_width=((0, max_target_length - len(item[1])), (0, 0)),
            mode='constant'
        )

        # padded_target = item[1]

        # for _ in range(max_target_length - len(item[1])):
        #     padded_target = np.append(
        #         padded_target,
        #         np.expand_dims(embeddings[''], axis=0),
        #         axis=0
        #     )

        padded_pw = np.pad(
            item[2],
            pad_width=(0, max_sentence_length - len(item[0])),
            mode='constant'
        )

        # print(len(padded_pw))

        padded_sentence_batch.append(padded_sentence)
        padded_target_batch.append(padded_target)
        padded_pw_batch.append(padded_pw)
        labels.append(item[3])

    return (
        padded_sentence_batch,
        padded_target_batch,
        padded_pw_batch,
        labels
    )

def get_normalized_batch(batch, embeddings):
    transformed_batch = [
        sentence_to_embeddings(sentence, target, pw, polarity, embeddings) 
        for sentence, target, pw, polarity in batch
    ]

    # get sequence length
    sentence_length, target_length = get_sequence_length(batch)
    # sentence_max_length, target_max_length = get_max_sequence_length(batch)

    # padding
    padded_batch = batch_embeddings_padding(transformed_batch, embeddings)

    return {
        'sentence_embeddings': np.array(padded_batch[0], dtype=np.float32),
        'target_embeddings': np.array(padded_batch[1], dtype=np.float32),
        'sentence_length': sentence_length,
        'target_length': target_length,
        'pw': np.array(padded_batch[2], dtype=np.float32),
        'labels': np.array(padded_batch[3], dtype=np.int32)
    }


    