import csv

DATA_PATH = '/home/skrhakv/cryptoshow-analysis/data/A-cluster-ligysis-data/clustered-binding-sites.txt'

def read_test_binding_residues(data_path=DATA_PATH, pocket_types=['CRYPTIC']) -> set[int]:
    cryptic_binding_residues = {}
    sequences = {}

    with open(data_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            chain_id = row[0][4:]
            protein_id = f'{row[0][:4]}_{chain_id}'
            sequence = row[4]

            if row[3] == '':
                continue

            binding_residue_indices = [f'{chain_id}_{int(i[1:])}'for i in row[3].split(' ')]
            if row[2] in pocket_types:
                if protein_id not in cryptic_binding_residues:
                    cryptic_binding_residues[protein_id] = []
                cryptic_binding_residues[protein_id].append(binding_residue_indices)
            sequences[protein_id] = sequence

    return cryptic_binding_residues, sequences
