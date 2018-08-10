import re
import requests

import pandas as pd

drug_target_regex = re.compile('(?:.|\n)*<td>Target uniprot</td><td>((.|\n)+)</td>(?:.|\n)*')
drug_drug_regex = re.compile('(?:.|\n)*([0-9]+) (?:interaction|interactions) with(?:.|\n)*')
UNIPROT_URL = 'https://www.uniprot.org/uniprot/'


def _read_entity2type(entity2type_path):
    with open(entity2type_path, 'r') as f:
        content = f.readlines()
    entity2type = {l.split(' ')[0].strip(): l.split(' ')[1].strip() for l in content}
    return entity2type


def _read_protein_interactions(protein_inter_file_path):
    with open(protein_inter_file_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip().split('\t') for line in lines]
        inter = {}
        for prot in lines:
            if prot[0] not in inter:
                inter[prot[0]] = [prot[1]]
            else:
                inter[prot[0]].append(prot[1])

            if prot[1] not in inter:
                inter[prot[1]] = [prot[0]]
            else:
                inter[prot[1]].append(prot[0])
    return inter


def _read_drugbank_interactions(drugbank_file_path):
    with open(drugbank_file_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip().split() for line in lines]
        inter = {}
        for ent in lines:
            source = ent[0]
            if source not in inter:
                inter[source] = []
            for e in ent[1:]:
                inter[source].append(e)
    return inter


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


def chain_interaction(ent_chain, inter):
    interaction = []
    for ent_1, ent_2 in zip(ent_chain[:-1], ent_chain[1:]):
        if ent_1 in inter and ent_2 in inter[ent_1]:
            interaction.append(1)
        elif ent_2 in inter and ent_1 in inter[ent_2]:
            interaction.append(1)
        else:
            if ent_1 not in inter and ent_2 not in inter:
                print(ent_1, ent_2)
            interaction.append(0)
    return interaction


if __name__ == '__main__':
    df = pd.read_json('./data/sentwise=F_cutoff=4_limit=500_method=all_tokenizer=punkt_medhop_train.json')
    inter_prot = _read_protein_interactions('./utils/reactome_2017-01-18_homo_sapiens_interactions.tsv')
    inter_drugbank = _read_drugbank_interactions('./utils/drugbank.tsv')

    inter = dict(inter_prot)
    inter.update(inter_drugbank)

    df['interactions'] = df.apply(lambda row: [chain_interaction(ch, inter) for ch in row['entity_paths']], axis=1)
    df.to_json('train_interactions_500')
