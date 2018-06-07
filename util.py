def get_target_relations_vocab(target_relations):
    return {rel: i for i, rel in enumerate(set(target_relations))}
