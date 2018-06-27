from path_rnn.nn.path_rnn import PathRNN


class ModelParameters:

    def __init__(self,
                 max_path_length=5,
                 max_relation_length=500,
                 path_rnn_hidden_size=150,
                 path_keep_prob=0.9,
                 aggregator=PathRNN.LOG_SUM_EXP,
                 relation_only=False,
                 recurrent_relation_embedder=False,
                 relation_rnn_hidden_size=150,
                 relation_keep_prob=0.9,
                 truncate_documents=False,
                 label_smoothing=0.0,
                 use_entity_type=False,
                 entity_type_embd_size=50):
        self.max_path_length = max_path_length
        self.max_relation_length = max_relation_length
        self.path_rnn_hidden_size = path_rnn_hidden_size
        self.path_keep_prob = path_keep_prob
        self.aggregator = aggregator
        self.relation_only = relation_only
        self.recurrent_relation_embedder = recurrent_relation_embedder
        self.relation_keep_prob = relation_keep_prob
        self.relation_rnn_hidden_size = relation_rnn_hidden_size
        self.truncate_documents = truncate_documents
        self.label_smoothing = label_smoothing
        self.use_entity_type = use_entity_type
        self.entity_type_embd_size = entity_type_embd_size

    def print(self):
        print('\n'.join("%s: %s" % item for item in vars(self).items()))
