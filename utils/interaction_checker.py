import pickle
import re
import requests
import time

import pandas as pd

from tqdm import tqdm

drug_target_regex = re.compile('(?:.|\n)*<td>Target uniprot</td><td>((.|\n)+)</td>(?:.|\n)*')
drug_drug_regex = re.compile('(?:.|\n)*([0-9]+) (?:interaction|interactions) with(?:.|\n)*')
UNIPROT_URL = 'https://www.uniprot.org/uniprot/'


def _read_entity2type(entity2type_path):
    with open(entity2type_path, 'r') as f:
        content = f.readlines()
    entity2type = {l.split(' ')[0].strip(): l.split(' ')[1].strip() for l in content}
    return entity2type


def protein_protein_interaction(protein_1, protein_2):
    response = requests.get(url=UNIPROT_URL, params={'query': 'interactor:{}'.format(protein_1),
                                                     'columns': 'id',
                                                     'format': 'tab'})
    interactors = response.content.decode('UTF-8').splitlines()[1:]
    return protein_2 in interactors


def drug_protein_interaction(drug, protein):
    response = requests.post('https://www.drugbank.ca/unearth/advanced/drugs',
                             'utf8=%E2%9C%93&authenticity_token=QJqsqDocCuJ7Kav8PM1kyZTN8eTSV%2BjyIzbNB%2Fl4MhNT6WbGjQ%'
                             '2BCh666F6xcG5vEJh38TGjk6tmiQVsn0xma%2FQ%3D%3D&commit=true&query_type=all&fields%5B1532794'
                             '691%5D%5Bfield%5D=drugbank_id&fields%5B1532794691%5D%5Bpredicate%5D=cont&fields%5B1532794'
                             '691%5D%5Bvalue%5D={}&display_fields%5B1532794671%5D%5Bfield%5D'
                             '=target_uniprot_id&button='.format(drug))
    match = drug_target_regex.match(response.content.decode('UTF-8'))
    if match:
        interactors = match.group(1)
        return protein in interactors.split(', ')
    else:
        return False


def drug_drug_interaction(drug_1, drug_2):
    response = requests.post('https://www.drugbank.ca/interax/multi_search',
                             'utf8=%E2%9C%93&authenticity_token=cavEmsV%2Fr%2BjNDe7BpJxjQ61D5eAmZ1QTnrPjUj7RRyRi2A70cm'
                             'wnjRieUpHESpxOH5PoSJzUVjgfxHVyFLDvyg%3D%3D&q={}%3B+{}&button='.format(drug_1, drug_2))

    match = drug_drug_regex.match(response.content.decode('UTF-8'))
    if match:
        interaction_count = int(match.group(1))
        return interaction_count > 0
    else:
        return False


def entity_entity_interaction(ent_1, ent_2, entity2type):
    interaction = False
    if entity2type[ent_1] == 'drug' and entity2type[ent_2] == 'protein':
        interaction = drug_protein_interaction(ent_1, ent_2)
    elif entity2type[ent_1] == 'protein' and entity2type[ent_2] == 'drug':
        interaction = drug_protein_interaction(ent_2, ent_1)
    elif entity2type[ent_1] == 'drug' and entity2type[ent_2] == 'drug':
        interaction = drug_drug_interaction(ent_1, ent_2)
    elif entity2type[ent_1] == 'protein' and entity2type[ent_2] == 'protein':
        interaction = protein_protein_interaction(ent_1, ent_2)
    return interaction


def chain_interaction(ent_chain, entity2type):
    try:
        interaction = []
        for ent_1, ent_2 in zip(ent_chain[:-1], ent_chain[1:]):
            interaction.append(int(entity_entity_interaction(ent_1, ent_2, entity2type)))
        time.sleep(1)
        return interaction
    except Exception:
        return [-1]


if __name__ == '__main__':
    print(drug_protein_interaction('DB00072', 'P11511'))
    print(drug_protein_interaction('DB00072', 'Meh'))
    print(drug_protein_interaction('Meh', 'Meh'))

    print(drug_drug_interaction('DB00333', 'DB00082'))

    print(protein_protein_interaction('P00520', 'Q99M51'))
    print(protein_protein_interaction('A', 'B'))

    entity2type = _read_entity2type('./parsing/entity_map.txt')
    print(chain_interaction(['DB08879', 'Q9Y275', 'P16410', 'DB06186'], entity2type))

    df = pd.read_json('.//data/sentwise=F_cutoff=4_limit=100_method=shortest_tokenizer=genia_medhop_train.json')
    i = []
    pickle.dump(i, open('train_interactions', 'wb'))
    for index, row in tqdm(df.iterrows()):
        interactions = [chain_interaction(ch, entity2type) for ch in row['entity_paths']]
        i.append(interactions)
    pickle.dump(i, open('train_interactions', 'wb'))
