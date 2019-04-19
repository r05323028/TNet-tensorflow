#! /bin/sh
GLOVE_NAME=glove.840B.300d
DATASET=14semeval_laptop

wget --no-check-certificate http://nlp.stanford.edu/data/${GLOVE_NAME}.zip
unzip ${GLOVE_NAME}.zip
mv ${GLOVE_NAME}.txt embeddings/${GLOVE_NAME}.txt
rm ${GLOVE_NAME}.zip

wget --no-check-certificate https://raw.githubusercontent.com/lixin4ever/TNet/master/dataset/${DATASET}/train.txt
wget --no-check-certificate https://raw.githubusercontent.com/lixin4ever/TNet/master/dataset/${DATASET}/test.txt
mv train.txt datasets/${DATASET}/train.txt
mv test.txt datasets/${DATASET}/test.txt