from functools import reduce


class DocumentStore:
    def __init__(self, medhop_instances):
        self.documents = [document_medhop_instances[0][0]
                          for document_medhop_instances
                          in medhop_instances]
        self.document_entities = []
        self.max_tokens = reduce(lambda p1, p2: max(p1, len(p2)), self.documents, 0)
        for idx, document_medhop_instances in enumerate(medhop_instances):
            entities_list = document_medhop_instances[0][1]
            self.document_entities.append({entity[0]: entity[1] for entity in entities_list})

    def _replace_entities(self, tokens, list_ent1, list_ent2, replacement_1, replacement_2):
        tokens_cpy = tokens.copy()
        for idx in list_ent1:
            tokens_cpy[idx] = replacement_1
        for idx in list_ent2:
            tokens_cpy[idx] = replacement_2
        return tokens_cpy

    def get_document(self, idx, source, target, truncate=False, replacement=None):
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




