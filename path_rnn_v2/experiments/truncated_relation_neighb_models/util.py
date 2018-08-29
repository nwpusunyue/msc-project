def max_entity_mentions(document_store):
    max = 0
    for doc_entities in document_store.document_entities:
        for pos in doc_entities.values():
            if len(pos) > max:
                max = len(pos)
    return max

def mean_entity_mentions(document_store):
    mentions = 0
    count = 0
    for doc_entities in document_store.document_entities:
        for pos in doc_entities.values():
            mentions += len(pos)
            count += 1
    return mentions / count


if __name__=='__main__':
    import pickle

    doc_store = pickle.load(open('/home/scimpian/msc-project/data/train_doc_store_genia.pickle', 'rb'))
    print(mean_entity_mentions(document_store=doc_store))
