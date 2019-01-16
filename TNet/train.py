import os
import numpy as np

from TNet.utils.embeddings import Glove
from TNet.utils.data import Batch
from TNet.utils import get_normalized_batch
from TNet.model import TNet
from tqdm import tqdm


if __name__ == "__main__":
    training_data_fname = 'dataset/14semeval_laptop/train.txt'
    testing_data_fname = 'dataset/14semeval_laptop/test.txt'
    embeddings_fname = 'embeddings/glove.840B.300d.txt'

    # rm log
    os.system(
        'rm -rf log/*'
    )

    # load data
    embeddings = Glove(embeddings_fname)
    batch_generator = Batch(training_data_fname)
    test_batch_generator = Batch(testing_data_fname)

    model = TNet()
    highest_acc = 0
    epoch = tqdm(range(0, 100), desc='epoch')

    for _ in epoch:
        acc_list = []
        for batch in batch_generator():
            feed_batch = get_normalized_batch(batch, embeddings)

            model.train_on_batch(**feed_batch)

        for batch in test_batch_generator():
            feed_test_batch = get_normalized_batch(batch, embeddings)
            test_acc = model.test_acc(**feed_test_batch)
            acc_list.append(test_acc)

        if np.mean(acc_list) > highest_acc:
            model.save_model()
            highest_acc = np.mean(acc_list)
            epoch.set_description('highest acc: {acc:.3f}'.format(acc=highest_acc))