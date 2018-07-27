from gensim.models import Word2Vec


def read_lines(file_obj):
    while True:
        try:
            line = file_obj.readline().strip()
            if not line:
                break
            if line[:2] == "b'" or line[:2] == 'b"':
                line = line[2:-1].replace('\\n', '\n')
                line = line.split('\n')
                for l in line:
                    l = l.split(' ')
                    yield l
            else:
                line = line.split(' ')
                yield line
        except UnicodeDecodeError:
            pass


class SentenceCorpus(object):

    def __init__(self, file_name):
        self.file_name = file_name

    def __iter__(self):
        count = 0
        with open(self.file_name, 'r', encoding='utf-8') as f:
            for l in read_lines(f):
                count += 1
                if count % 1000000 == 0:
                    print(count, end='...', flush=True)
                yield l


model = Word2Vec(min_count=5)
model.build_vocab(SentenceCorpus('./medline.txt'))
print('Vocab built. Size: {}'.format(len(model.wv.vocab)))
model.train(SentenceCorpus('./medline.txt'), total_examples=model.corpus_count, epochs=15)
model.save('./medline_word2vec')
