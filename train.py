import os
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from configparser import ConfigParser

from TNet.utils.embeddings import Glove
from TNet.utils.data import Batch
from TNet.utils import get_normalized_batch
from TNet.model import TNet


def get_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--mode', type=str, default='as')
    arg_parser.add_argument('--train_data_fp', type=str, default='datasets/14semeval_laptop/train.txt')
    arg_parser.add_argument('--test_data_fp', type=str, default='datasets/14semeval_laptop/test.txt')
    arg_parser.add_argument('--embedding_fp', type=str, default='embeddings/glove.840B.300d.txt')
    arg_parser.add_argument('--hparams_fp', type=str, default='./hparams.ini')
    arg_parser.add_argument('--model_name', type=str, default='tnet')
    args = arg_parser.parse_args()

    return vars(args)

def load_hparams(fp):
    conf_parser = ConfigParser()
    conf_parser.read(fp)

    return conf_parser

def clean_log():
    os.system(
        'rm -rf log/*'
    )

def main(**kwargs):
    training_data_fname = kwargs.get('train_data_fp')
    testing_data_fname = kwargs.get('test_data_fp')
    embeddings_fname = kwargs.get('embedding_fp')

    # rm log
    clean_log()

    hparams = load_hparams(kwargs.get('hparams_fp'))

    # load data
    embeddings = Glove(embeddings_fname)
    batch_generator = Batch(training_data_fname, hparams, mode=kwargs.get('mode'), shuffle=True)
    test_batch_generator = Batch(testing_data_fname, hparams, mode=kwargs.get('mode'))

    model = TNet(hparams, **kwargs)
    epoch = tqdm(range(0, int(hparams['global']['num_epochs'])), desc='epoch')
    highest_acc = 0

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
            epoch.set_description('highest test acc: {acc:.3f}'.format(acc=highest_acc))


if __name__ == "__main__":
    args = get_args()
    main(**args)