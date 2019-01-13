# Transformation Network (TensorFlow)

This repo is the implementation of Transformation Networks for Target-Oriented Sentiment Classification (Xin Li, Lidong Bing, Wai Lam, Bei Shi, 2018) which is accepted by ACL 2018.

But, it still has some problems and can't reach the reported results. Before training the model, you need to get embedding and dataset.

- Glove embedding: https://nlp.stanford.edu/projects/glove/
- Dataset: https://github.com/lixin4ever/TNet

|Model|Dataset|Accuracy|
|---|---|---|
|Paper|LAPTOP|76.54%|
|tensorflow|LAPTOP|~65%|

## Requirements

```
pip install -r requirements.txt
```

## Train Model

```
python -m sentiment.train
```

## Training Monitor

```
tensorboard --logdir=log --host=<host> --port=<port>
```

## Reference
- <a href='https://ai.tencent.com/ailab/media/publications/acl/Transformation_Networks_for_Target-Oriented_Sentiment_Classification.pdf'>Transformation Networks for Target-Oriented Sentiment Classification</a>