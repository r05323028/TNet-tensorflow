import os
import numpy as np

from TNet.utils.embeddings import Glove
from TNet.utils.data import Batch
from TNet.utils import get_normalized_batch
from TNet.model import TNet
from tqdm import tqdm


if __name__ == "__main__":
    training_data_fname = '/home/sean/ds-sentiment-analysis/TNet/dataset/14semeval_laptop/train.txt'
    embeddings_fname = '/home/sean/ds-sentiment-analysis/TNet/embeddings/glove.840B.300d.txt'

    # rm log
    os.system(
        'rm -rf log/*'
    )

    # load data
    embeddings = Glove(embeddings_fname)
    batch_generator = Batch(training_data_fname)

    # load model
    model = TNet()

    epoch = tqdm(range(0, 100), desc='epoch')

    for _ in epoch:
        for batch in batch_generator():
            feed_batch = get_normalized_batch(batch, embeddings)

            model.train_on_batch(**feed_batch)

    model.save_model()