import nltk
import os

import numpy as np
import pandas as pd

from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import tree2conlltags


def text_to_atoms_ner(text, tokenizer=None):
    """
    Uses NER tags to identify more coarse grained entities than text_to_atoms_POS()
    "binarize" flag: whether to convert n-ary tuples into collection of 2-ary tuples.
    """
    if tokenizer is None:
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    sentences = tokenizer.tokenize(text)

    atoms = []
    for sentence in sentences:

        tokens = word_tokenize(sentence)

        # get NER tags on sentence in BIO schema
        iob_tagged = tree2conlltags(ne_chunk(pos_tag(tokens)))

        # remove anything labelled as 'O'
        chunks = [tup for tup in iob_tagged if tup[2] != 'O']

        entities = []
        currend_id = -1

        for chunk in chunks:
            if chunk[2][0] == 'B':
                currend_id += 1
                entities.append(chunk[0])
            elif chunk[2][0] == 'I':
                # concatenate current token to previous token (I = inside the tag)
                entities[-1] = entities[-1] + " " + chunk[0]

            current_token_position = tokens.index(chunk[0])
            if tokens[current_token_position - 1] == '###{}###'.format(currend_id):
                # collapse multi-token entities into single placeholder token
                tokens[current_token_position] = ""
            else:
                tokens[tokens.index(chunk[0])] = '###{}###'.format(currend_id)

        arguments = entities

        if not arguments:
            continue

        atoms += arguments

    return atoms


def text_to_atoms_pos(text, tokenizer=None):
    if tokenizer is None:
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    sentences = tokenizer.tokenize(text)
    atoms = []
    for sentence in sentences:
        # List of arguments
        arguments = []
        # Tokenizing each sentence
        tokens = nltk.word_tokenize(sentence)
        # POS-tagging every token in the sentence
        _token_pos_lst = nltk.pos_tag(tokens)

        # Pre-processing step: aggregate NNP (noun phrase) tokens together
        token_pos_lst = []

        nnp_token = None
        nnps_token = None

        for token_pos in _token_pos_lst:
            token, pos = token_pos
            # If it's a NNP ..
            if pos in {'NNP'}:
                # Aggregate it with previous NNPs in the sequence
                nnp_token = '{} {}'.format(nnp_token, token) if nnp_token else token

            elif pos in {'NNPS'}:
                nnps_token = '{} {}'.format(nnps_token, token) if nnps_token else token

            # Otherwise ..
            else:
                # If there was a sequence of NNP tokens before ..
                if nnp_token:
                    # Add such tokens to the list as a single token
                    token_pos_lst += [(nnp_token, 'NNP')]
                    nnp_token = None

                if nnps_token:
                    token_pos_lst += [(nnps_token, 'NNPS')]
                    nnps_token = None

                # And then add the current token as well
                token_pos_lst += [token_pos]

        if nnp_token:
            token_pos_lst += [(nnp_token, 'NNP')]

        if nnps_token:
            token_pos_lst += [(nnps_token, 'NNPS')]

        for token, pos in token_pos_lst:
            if pos in {'NN', 'NNS', 'NNP', 'NNPS'}:
                arguments += [token]  # e.g. Obama
        atoms += arguments
    return atoms


class OverlappingEntities:

    def __init__(self):
        self.ent = []
        self.doc2ent = {}

    def add_entity(self, doc_id, new_entity):
        new_entity_set = set(new_entity)
        appended = False
        for ent_id, ent in enumerate(self.ent):
            max_overlap = 0
            max_tok = 0
            for form in ent:
                overlap = len(set(form) & new_entity_set)
                if overlap > max_overlap:
                    max_overlap = overlap
                if len(form) > max_tok:
                    max_tok = len(form)
            if max_overlap > 0.5 * max(max_tok, len(new_entity)):
                self.ent[ent_id].append(new_entity)
                new_ent_id = ent_id
                appended = True
        if not appended:
            new_ent_id = len(self.ent)
            self.ent.append([new_entity])
        if doc_id not in self.doc2ent:
            self.doc2ent[doc_id] = []
        if new_ent_id not in self.doc2ent[doc_id]:
            self.doc2ent[doc_id].append(new_ent_id)

    def get_max_overlap_ent(self, target):
        target_set = set(target)
        global_max_overlap = 0
        max_ent_id = 0
        for ent_id, ent in enumerate(self.ent):
            max_overlap = 0
            max_tok = 0
            for form in ent:
                overlap = len(set(form) & target_set)
                if overlap > max_overlap:
                    max_overlap = overlap
                if len(form) > max_tok:
                    max_tok = len(form)
            if max_overlap > global_max_overlap:
                global_max_overlap = max_overlap
                max_ent_id = ent_id
        print('Max overlap: {}'.format(global_max_overlap))
        if global_max_overlap == 0:
            return None
        else:
            return max_ent_id

    def get_graph_info(self):
        adj = np.zeros((len(self.ent), len(self.ent)))
        labels = {}
        rel = {}
        for doc_id, doc_ents in self.doc2ent.items():
            for i in range(len(doc_ents)):
                if doc_ents[i] not in labels:
                    labels[doc_ents[i]] = self.ent[doc_ents[i]]
                for j in range(i + 1, len(doc_ents)):
                    ent_1 = doc_ents[i]
                    ent_2 = doc_ents[j]
                    adj[ent_1, ent_2] = 1.0
                    adj[ent_1, ent_2] = 1.0
                    if (ent_1, ent_2) not in rel:
                        rel[(ent_1, ent_2)] = [doc_id]
                        rel[(ent_2, ent_1)] = [doc_id]
                    else:
                        rel[(ent_1, ent_2)].append(doc_id)
                        rel[(ent_2, ent_1)].append(doc_id)
        return adj, rel, labels


def get_entities(supports):
    support_atoms = []
    for support in supports:
        support_atoms.append(text_to_atoms_pos(support))

    overlapping_entities = OverlappingEntities()
    for sup_id, atoms in enumerate(support_atoms):
        for atom in atoms:
            overlapping_entities.add_entity(sup_id, atom.lower().split(' '))
    return overlapping_entities


if __name__ == '__main__':
    print(os.getcwd())
    df = pd.read_json('./../../qangaroo_v1.1/wikihop/train.json', orient='records')
    df['entities'] = df.apply(lambda row: get_entities(row['supports']), axis=1)
    print(df['entities'][0].ent)

