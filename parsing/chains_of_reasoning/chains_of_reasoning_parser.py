import argparse
import json
import os

# Instantiate the parser
parser = argparse.ArgumentParser()
parser.add_argument('--relation',
                    type=str,
                    help='Relation type')


def parse_dataset(relation, dataset_type):
    with open('/cluster/project2/mr/scimpian/chains_of_reasoning_data/{}/{}_matrix.tsv.translated'.format(relation,
                                                                                                          dataset_type),
              'r') as f:
        lines = f.readlines()
        if dataset_type in ['positive', 'negative']:
            if dataset_type == 'positive':
                label = '1'
            else:
                label = '-1'
            lines = [line.split('\t') + [label] for line in lines]
        else:
            lines = [line.split('\t') for line in lines]
        lines = [(line[0].strip(), line[1].strip(), line[2].strip().split('###'), line[3].strip()) for line in lines]
        lines = [(line[0], line[1], [rel.split('-') for rel in line[2]], (int(line[3]) + 1) / 2) for line in lines]

        lines = [(line[0], line[1], [(path[0::2] + ['#END'],
                                      [line[0]] + path[1::2] + [line[1]]) for path in line[2]], line[3]) for line in
                 lines]

        dataset = {'source': [],
                   'target': [],
                   'relation_paths': [],
                   'entity_paths': [],
                   'label': [],
                   'target_relation': []}

        for l in lines:
            dataset['source'].append(l[0])
            dataset['target'].append(l[1])
            dataset['relation_paths'].append([p[0] for p in l[2]])
            dataset['entity_paths'].append([p[1] for p in l[2]])
            dataset['target_relation'].append('/people/person/nationality')
            dataset['label'].append(l[3])

        if 'parsed' not in os.listdir('/cluster/project2/mr/scimpian/chains_of_reasoning_data/{}'.format(relation)):
            os.mkdir('/cluster/project2/mr/scimpian/chains_of_reasoning_data/{}/parsed'.format(relation))
        with open('/cluster/project2/mr/scimpian/chains_of_reasoning_data/{}'
                  '/parsed/{}_matrix.json'.format(relation,
                                                  dataset_type),
                  'w+') as w:
            w.write(json.dumps(dataset))


if __name__ == '__main__':
    args = parser.parse_args()
    relation = args.relation
    print('Relation: {}'.format(relation))
    for dataset_type in ['positive', 'negative', 'dev', 'test']:
        print('Generating {}'.format(dataset_type))
        parse_dataset(relation, dataset_type)
        print('Done.')
