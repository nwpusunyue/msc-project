from functools import reduce
from itertools import product


class DocumentStore:
    def __init__(self, medhop_instances):
        self.documents = [document_medhop_instances[0][0]
                          for document_medhop_instances
                          in medhop_instances]
        self.document_sentences = [[(sent_medhop_instances[0], {ent[0]: ent[1] for ent in sent_medhop_instances[1]})
                                    for sent_medhop_instances
                                    in document_medhop_instances[0][3]
                                    ]
                                   for document_medhop_instances
                                   in medhop_instances]

        self.document_entities = []
        self.document_sentence_entities = []

        self.max_tokens = reduce(lambda p1, p2: max(p1, len(p2)), self.documents, 0)

        for idx, document_medhop_instances in enumerate(medhop_instances):
            entities_list = document_medhop_instances[0][1]
            sentence_entities_list = document_medhop_instances[0][2]
            self.document_entities.append({entity[0]: entity[1] for entity in entities_list})
            self.document_sentence_entities.append(sentence_entities_list)

    def _replace_entities(self, tokens, list_ent1, list_ent2, replacement_1, replacement_2):
        tokens_cpy = tokens.copy()
        for idx in list_ent1:
            tokens_cpy[idx] = replacement_1
        for idx in list_ent2:
            tokens_cpy[idx] = replacement_2
        return tokens_cpy

    def get_document(self, idx, source, target, truncate=False, sentence_truncate=False, replacement=None):
        if sentence_truncate:
            return self.get_sentences(idx, source, target, replacement)

        if truncate:
            pos = self.document_entities[idx][source] + self.document_entities[idx][target]
            start = min(pos)
            end = max(pos) + 1
        else:
            start = 0
            end = len(self.documents[idx])

        if replacement is None:
            replacement_1 = source
            replacement_2 = target
        else:
            replacement_1, replacement_2 = replacement

        return self._replace_entities(self.documents[idx],
                                      self.document_entities[idx][source],
                                      self.document_entities[idx][target],
                                      replacement_1=replacement_1,
                                      replacement_2=replacement_2)[start:end]

    def get_sentences(self, idx, source, target, replacement=None):
        if replacement is None:
            replacement_1 = source
            replacement_2 = target
        else:
            replacement_1, replacement_2 = replacement

        doc_sent = self.document_sentences[idx]
        doc_sent_entities = self.document_sentence_entities[idx]

        source_sent_idx = doc_sent_entities[source]
        target_sent_idx = doc_sent_entities[target]

        all_sent_idx = sorted(set(source_sent_idx + target_sent_idx))

        context = []
        for sent_idx in all_sent_idx:
            sent = doc_sent[sent_idx]
            source_sent_pos = sent[1][source] if source in sent[1] else []
            target_sent_pos = sent[1][target] if target in sent[1] else []
            context += self._replace_entities(sent[0],
                                              source_sent_pos,
                                              target_sent_pos,
                                              replacement_1=replacement_1,
                                              replacement_2=replacement_2)
        return context

    def get_entity_neighb(self, idx, ent, neighb_size=0, replacement=None):
        if neighb_size == 0:
            return [ent if replacement is None else replacement]

        doc = self.documents[idx]
        ent_pos = self.document_entities[idx][ent]
        ent_neighb = []

        for ep in ent_pos:
            start = max(0, ep - neighb_size)
            end = min(len(doc), ep + neighb_size + 1)

            for p in range(start, end):
                if p == ep:
                    if replacement is not None:
                        ent_neighb.append(replacement)
                    else:
                        ent_neighb.append(ent)
                else:
                    ent_neighb.append(doc[p])

        return ent_neighb

    def get_min_distance(self, idx, source, target):
        pos_source = self.document_entities[idx][source]
        pos_target = self.document_entities[idx][target]
        return min([abs(p[0] - p[1]) for p in product(pos_source, pos_target)])
