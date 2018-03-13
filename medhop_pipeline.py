import re
import pandas as pd
from itertools import combinations
from nltk.tokenize import sent_tokenize, word_tokenize


def extract_entities(sentence):
    biomed_entity = re.compile("[ABCDEFGHIKMOPQS][0-9,A-Z]+[0-9]+")
    unique_entities = {}

    for i, t in enumerate(word_tokenize(sentence)):
        if biomed_entity.match(t):
            if t not in unique_entities:
                unique_entities[t] = [i]
            else:
                unique_entities[t].append(i)
    # return list of (entity, [list_of_occurences_indices]) tuple, sorted by the first occurence index
    return sorted([(k, v) for k, v in unique_entities.items()], key=lambda x: x[1][0])


def extract_medhop_instances(documents):
    instances = []
    for d in documents:
        documents_instances = list(filter(lambda x: len(x[1]) > 1,
                                          [(s, extract_entities(s)) for s in sent_tokenize(d)]))
        instances += documents_instances
    return instances


def extract_binary_medhop_instances(medhop_instances):
    binary_instances = []
    for instance in medhop_instances:
        pairs = list(combinations(instance[1], 2))
        binary_instances += [(instance[0], pair) for pair in pairs]
    return binary_instances


def preprocess_medhop(filename):
    df = pd.read_json(filename, orient='records')
    df['medhop_instances'] = df['supports'].apply(extract_medhop_instances)
    df['binary_medhop_instances'] = df['medhop_instances'].apply(extract_binary_medhop_instances)
    return df


if __name__ == "__main__":
    df = preprocess_medhop('./qangaroo_v1.1/medhop/train.json')
    print(df['binary_medhop_instances'])
