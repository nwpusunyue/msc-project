from functools import reduce
from parsing.special_tokens import *


class DocumentStore:
    def __init__(self, medhop_instances):
        self.documents = [document_medhop_instances[0][0]
                          for document_medhop_instances
                          in medhop_instances]
        self.document_entities = []
        self.max_tokens = reduce(lambda p1, p2: max(p1, len(p2)),  self.documents, 0)
        for idx, document_medhop_instances in enumerate(medhop_instances):
            entities_list = document_medhop_instances[0][1]
            self.document_entities.append({entity[0]: entity[1] for entity in entities_list})

    def _replace_entities(self, tokens, list_ent1, list_ent2):
        tokens_cpy = tokens.copy()
        for idx in list_ent1:
            tokens_cpy[idx] = ENT_1
        for idx in list_ent2:
            tokens_cpy[idx] = ENT_2
        return tokens_cpy

    def get_document(self, idx, source, target):
        return self._replace_entities(self.documents[idx],
                                      self.document_entities[idx][source],
                                      self.document_entities[idx][target])
