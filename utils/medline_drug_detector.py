from tqdm import tqdm


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


def find_drug_mentions(medline_path, drug_path, new_file_path):
    corpus = SentenceCorpus(medline_path)
    drugs = open(drug_path, 'r').readlines()
    drugs = [d.strip() for d in drugs]

    with open(new_file_path, 'w') as f:
        for sent in tqdm(corpus):
            mentions = set()
            for token in sent:
                if token in drugs:
                    mentions.add(token)
            if len(mentions) >= 2:
                f.write(' '.join(sent))
                f.write('\n')
                f.flush()


if __name__ == '__main__':
    find_drug_mentions('/cluster/project2/mr/scimpian/medline.txt', '/home/scimpian/msc-project/parsing/drugs.txt',
                       '/cluster/project2/mr/scimpian/medline_drugs_sent.txt')
