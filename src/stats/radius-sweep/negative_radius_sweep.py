import csv
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.utils import class_weight
from torch.utils.data import DataLoader, Dataset

PROJECT_DIRECTORY = '/home/skrhakv/Projects/seq2pocket'
CRYPTIC_NN_DATA = '/work/skrhakv/cryptic-nn'
sys.path.append(f'{PROJECT_DIRECTORY}/src/utils')

from eval_utils import CryptoBenchClassifier  # noqa: E402  (dim=2048, dropout=0.5 by default)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMBEDDINGS_DIR = f'{CRYPTIC_NN_DATA}/embeddings'
DISTANCE_MATRICES_DIR = f'{CRYPTIC_NN_DATA}/distance-matrices'
TRAIN_PATH = f'{CRYPTIC_NN_DATA}/train_val.txt'
VAL_PATH = f'{CRYPTIC_NN_DATA}/val.txt'

POSITIVE_DISTANCE_THRESHOLD = 15  # candidate radius, fixed at the paper value
RADIUS_GRID = [10, 15]

LR = 1e-4
BATCH_SIZE = 512
EPOCHS = 150
EARLY_STOPPING_PATIENCE = 10
TEST_DECISION_THRESHOLD = 0.5


def process_sequence_dataset(annotation_path, negative_radius):
    Xs, Ys = {}, {}
    with open(annotation_path) as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            id_ = row[0].lower() + row[1]
            if row[3] == '':
                continue
            Xs[id_], Ys[id_] = [], []

            embedding = np.load(f'{EMBEDDINGS_DIR}/{id_}.npy')
            distance_matrix = np.load(f'{DISTANCE_MATRICES_DIR}/{id_}.npy')

            binding_residues_indices = [int(r[1:]) for r in row[3].split(' ')]
            negative_examples_indices = set()

            for residue_idx in binding_residues_indices:
                close_residues_indices = np.where(distance_matrix[residue_idx] < POSITIVE_DISTANCE_THRESHOLD)[0]
                close_binding_residues_indices = np.intersect1d(close_residues_indices, binding_residues_indices)
                concatenated_embedding = np.concatenate(
                    (embedding[residue_idx], np.mean(embedding[close_binding_residues_indices], axis=0))
                )
                Xs[id_].append(concatenated_embedding)
                Ys[id_].append(1)

                really_close_residues_indices = np.where(distance_matrix[residue_idx] < negative_radius)[0]
                negative_examples_indices.update(set(really_close_residues_indices) - set(binding_residues_indices))

            for residue_idx in negative_examples_indices:
                close_residues_indices = np.where(distance_matrix[residue_idx] < POSITIVE_DISTANCE_THRESHOLD)[0]
                close_binding_residues_indices = np.intersect1d(close_residues_indices, binding_residues_indices)
                concatenated_embedding = np.concatenate(
                    (embedding[residue_idx], np.mean(embedding[close_binding_residues_indices], axis=0))
                )
                Xs[id_].append(concatenated_embedding)
                Ys[id_].append(0)

    return Xs, Ys


class SmoothnessClassifierDataset(Dataset):
    def __init__(self, Xs, Ys):
        Xs_list = np.concatenate(list(Xs.values()), axis=0)
        Ys_list = np.concatenate(list(Ys.values()), axis=0)
        self.Xs = torch.tensor(Xs_list, dtype=torch.float32)
        self.Ys = torch.tensor(Ys_list, dtype=torch.int64)

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        return self.Xs[idx], self.Ys[idx]


def train_and_select(train_dataset, val_dataset):
    model = CryptoBenchClassifier().to(DEVICE)  # dim=2048, dropout=0.5 (already-selected architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    _, y_train = train_dataset[:]
    X_val, y_val = val_dataset[:]
    X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE).float()

    class_weights = torch.tensor(
        class_weight.compute_class_weight(
            class_weight='balanced', classes=np.unique(y_train.numpy()), y=y_train.numpy()
        ),
        dtype=torch.float,
    ).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=2 * class_weights[1])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    best_auprc, best_mcc, best_f1, best_acc, best_epoch = 0, 0, 0, 0, 0
    early_stopping_counter = 0
    for epoch in range(EPOCHS):
        model.eval()
        with torch.inference_mode():
            val_logits = model(X_val).squeeze()
            val_probs = torch.sigmoid(val_logits)
            val_pred = (val_probs > TEST_DECISION_THRESHOLD).float()

            precision, recall, _ = metrics.precision_recall_curve(y_val.cpu().numpy(), val_probs.cpu().numpy())
            auprc = metrics.auc(recall, precision)

            if auprc > best_auprc:
                best_auprc = auprc
                best_mcc = metrics.matthews_corrcoef(y_val.cpu().numpy(), val_pred.cpu().numpy())
                best_f1 = metrics.f1_score(y_val.cpu().numpy(), val_pred.cpu().numpy(), average='weighted')
                best_acc = (torch.eq(y_val, val_pred).sum().item() / len(y_val)) * 100
                best_epoch = epoch
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                    break

        model.train()
        for x_batch, y_batch in train_dataloader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE).float()
            y_logits = model(x_batch).squeeze()
            loss = loss_fn(y_logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return best_auprc, best_mcc, best_f1, best_acc, best_epoch


def main():
    print(f'\n{"negative_radius":<18}{"AUPRC":<10}{"MCC":<10}')
    for negative_radius in RADIUS_GRID:
        torch.manual_seed(42)

        Xs_train, Ys_train = process_sequence_dataset(TRAIN_PATH, negative_radius)
        train_dataset = SmoothnessClassifierDataset(Xs_train, Ys_train)

        Xs_val, Ys_val = process_sequence_dataset(VAL_PATH, negative_radius)
        val_dataset = SmoothnessClassifierDataset(Xs_val, Ys_val)

        auprc, mcc, _, _, _ = train_and_select(train_dataset, val_dataset)
        print(f'{negative_radius:<18}{auprc:<10.4f}{mcc:<10.4f}')


if __name__ == '__main__':
    main()
