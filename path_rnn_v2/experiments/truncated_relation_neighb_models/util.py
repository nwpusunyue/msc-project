def max_entity_mentions(document_store):
    max = 0
    for doc_entities in document_store.document_entities:
        for pos in doc_entities.values():
            if len(pos) > max:
                max = len(pos)
    return max
