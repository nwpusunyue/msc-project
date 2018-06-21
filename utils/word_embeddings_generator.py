import argparse
import os

import pandas as pd

from gensim.models import Word2Vec
from itertools import chain
from nltk.tokenize import sent_tokenize
from nltk.tokenize import TreebankWordTokenizer, WordPunctTokenizer

tokenizer = {
    'punkt': WordPunctTokenizer(),
    'treebank': TreebankWordTokenizer()
}

# Instantiate the parser
parser = argparse.ArgumentParser()
parser.add_argument('train_path',
                    type=str,
                    help='Train dataset path')

parser.add_argument('dev_path',
                    type=str,
                    help='Dev dataset path')

parser.add_argument('output_path',
                    type=str,
                    help='Word2Vec model path')

parser.add_argument('--tokenizer',
                    type=str,
                    default='punkt',
                    help='One of punkt or treebank')

parser.add_argument('--epochs',
                    type=int,
                    default=15,
                    help='Epochs to train Word2Vec emb for')

parser.add_argument('--min_count',
                    type=int,
                    default=5,
                    help='Min num of occurrences of a word so that it gets an embedding ')


def generate_word2vec(documents, tokenizer_name, epochs, min_count, save_path):
    print('{} total documents'.format(len(documents)))
    sentences = set(chain.from_iterable(sent_tokenize(d) for d in documents))
    print('{} total sentences'.format(len(sentences)))

    tokenized_sentences = [tokenizer[tokenizer_name].tokenize(s) for s in sentences]
    model = Word2Vec(min_count=min_count)
    model.build_vocab(tokenized_sentences)
    print('{} vocab size'.format(len(model.wv.vocab)))
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)
    model.save(save_path)


args = parser.parse_args()

print(os.getcwd())
print('Train path: {}\n'
      'Dev path: {}\n'
      'Output path: {}\n'
      'Tokenizer: {}\n'
      'Epochs: {}\n'
      'Min count: {}\n'.format(args.train_path,
                               args.dev_path,
                               args.output_path,
                               args.tokenizer,
                               args.epochs,
                               args.min_count))

train_df = pd.read_json(args.train_path, orient='records')
dev_df = pd.read_json(args.dev_path, orient='records')
supports = set(chain.from_iterable(list(dev_df['supports']) + list(train_df['supports'])))
generate_word2vec(supports, args.tokenizer, args.epochs, args.min_count, args.output_path)
