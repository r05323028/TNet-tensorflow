# Transformation Networks (TensorFlow)

This repo is the implementation of Transformation Networks for Target-Oriented Sentiment Classification (Xin Li, Lidong Bing, Wai Lam, Bei Shi, 2018) which is accepted by ACL 2018.

Before training the model, you need to get the embedding and dataset.

- Glove embedding: https://nlp.stanford.edu/projects/glove/ (glove.840B.300d)
- Dataset: https://github.com/lixin4ever/TNet

or you can use `prepare_data.sh` to download the laptop dataset and the embedding automately.

```
. ./prepare_data.sh
```

## Changelog

- add the `prepare_data.sh` script.
- fix the performance problem (forget shuffle on each epoch.)

## Requirements

- Python 3.6.x
- tensorflow 1.12.0
- numpy 1.15.3
- tqdm 4.23.4

```
pip install -r requirements.txt
```

## Train Model

```
python train.py
```

## Training Monitor

```
tensorboard --logdir=log --host=<host> --port=<port>
```

## Reference
- <a href='https://ai.tencent.com/ailab/media/publications/acl/Transformation_Networks_for_Target-Oriented_Sentiment_Classification.pdf'>Transformation Networks for Target-Oriented Sentiment Classification</a>