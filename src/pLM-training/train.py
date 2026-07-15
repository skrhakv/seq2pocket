import torch
import numpy as np
import sys
import argparse
import functools
from transformers import AutoTokenizer
import torch.nn as nn
from sklearn import metrics
import gc
import bitsandbytes as bnb
from torch.utils.data import DataLoader
import warnings

sys.path.append('/home/skrhakv/Projects/cryptic-nn/src')
import baseline_utils
import finetuning_utils  # download from https://github.com/skrhakv/cryptic-finetuning/blob/master/src/finetuning_utils.py

torch.manual_seed(42)
warnings.filterwarnings('ignore')

MODEL_NAME = 'facebook/esm2_t36_3B_UR50D'
# MODEL_NAME = 'facebook/esm2_t33_650M_UR50D'
# MODEL_NAME = 'facebook/esm2_t6_8M_UR50D' # for testing purposes
DATA_DIRECTORY = '/home/skrhakv/Projects/seq2pocket/data'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train ESM2 binding site predictor. '
                    'PU learning flags address incomplete annotation: '
                    'some label-0 residues may be unobserved positives.'
    )
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs (default: 3)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--output', type=str,
                        default=f'{DATA_DIRECTORY}/models/gbs-model-enhanced-scPDB-filtered.pt',
                        help='Output model path')

    # --- Label smoothing on negatives ---
    ls = parser.add_argument_group('Label smoothing',
                                   'Soft-penalizes negative labels to account for unobserved positives.')
    ls.add_argument('--label-smoothing', action='store_true',
                    help='Apply label smoothing to negative (label-0) residues')
    ls.add_argument('--label-smoothing-alpha', type=float, default=0.05,
                    help='Target value for smoothed negatives, e.g. 0.05 means 0->0.05 (default: 0.05)')

    # --- nnPU loss ---
    pu = parser.add_argument_group('nnPU loss',
                                   'Treats label-0 residues as unlabeled (Kiryo et al. 2017). '
                                   'Cannot be combined with --label-smoothing.')
    pu.add_argument('--nnpu', action='store_true',
                    help='Use non-negative PU loss instead of BCE')
    pu.add_argument('--nnpu-prior', type=float, default=None,
                    help='Class prior π for nnPU. If not set, estimated from training labels.')

    # --- Self-training / pseudo-labeling ---
    pl = parser.add_argument_group('Pseudo-labeling',
                                   'Iteratively expands the positive set using the model\'s '
                                   'high-confidence predictions on unlabeled residues.')
    pl.add_argument('--pseudo-labels', action='store_true',
                    help='Enable self-training pseudo-labeling')
    pl.add_argument('--pseudo-label-threshold', type=float, default=0.9,
                    help='Sigmoid confidence threshold to promote a label-0 residue to positive (default: 0.9)')
    pl.add_argument('--pseudo-label-epoch', type=int, default=2,
                    help='Run pseudo-labeling before this epoch (default: 2, aligned with LLM unfreeze)')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def compute_nnpu_loss(logits, labels, prior):
    """
    Non-negative PU loss (Kiryo et al. NeurIPS 2017).

    Decomposes the risk into a positive term and an unlabeled term, then
    non-negatively clamps the unlabeled part to prevent overfitting to noise.

      R_nnpu = π·R_p⁺  +  max(0,  R_u⁻  −  π·R_p⁻)

    where p = confirmed positives, u = unlabeled (label-0), prior π ≈ P(y=1).
    """
    bce = nn.BCEWithLogitsLoss(reduction='none')
    pos_mask = labels == 1
    neg_mask = labels == 0  # "unlabeled" in PU terminology

    # Degenerate-batch guards
    if neg_mask.sum() == 0:
        return bce(logits[pos_mask], torch.ones_like(logits[pos_mask])).mean()
    if pos_mask.sum() == 0:
        return bce(logits[neg_mask], torch.zeros_like(logits[neg_mask])).mean()

    loss_pos_as_pos = bce(logits[pos_mask], torch.ones_like(logits[pos_mask])).mean()
    loss_pos_as_neg = bce(logits[pos_mask], torch.zeros_like(logits[pos_mask])).mean()
    loss_unl_as_neg = bce(logits[neg_mask], torch.zeros_like(logits[neg_mask])).mean()

    negative_risk = loss_unl_as_neg - prior * loss_pos_as_neg
    return prior * loss_pos_as_pos + torch.clamp(negative_risk, min=0.0)


def compute_train_loss(logits, labels, args, loss_fn, prior):
    """Dispatch to the appropriate loss based on CLI flags."""
    if args.nnpu:
        return compute_nnpu_loss(logits, labels, prior)

    if args.label_smoothing:
        smoothed = labels.clone().float()
        smoothed[labels == 0] = args.label_smoothing_alpha
        return loss_fn(logits, smoothed)

    return loss_fn(logits, labels.float())


# ---------------------------------------------------------------------------
# Pseudo-labeling
# ---------------------------------------------------------------------------

def run_pseudo_labeling(model, dataset, tokenizer, threshold, device):
    """
    One inference pass over the training set.

    Any residue currently labeled 0 whose sigmoid(logit) >= threshold is
    promoted to 1. Returns a new dataset with updated labels.
    """
    partial_collate = functools.partial(finetuning_utils.collate_fn, tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=partial_collate)

    model.eval()
    # Work on a plain Python copy so we can mutate freely
    new_labels = [list(row) for row in dataset['labels']]
    total_promoted = 0

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            output = model(batch)
            labels_tensor = batch['labels'].to(device)

            flat_logits = output.flatten()
            flat_labels = labels_tensor.flatten()
            valid_mask = flat_labels != -100  # strips BOS / EOS / padding positions

            valid_logits = flat_logits[valid_mask]
            valid_labels = flat_labels[valid_mask]
            probs = torch.sigmoid(valid_logits).cpu().float()

            assert len(probs) == len(new_labels[idx]), (
                f"Length mismatch at sample {idx}: "
                f"model output {len(probs)} vs label array {len(new_labels[idx])}"
            )

            for pos in range(len(valid_labels)):
                if valid_labels[pos].item() == 0 and probs[pos].item() >= threshold:
                    new_labels[idx][pos] = 1
                    total_promoted += 1

            del output, labels_tensor
            gc.collect()
            torch.cuda.empty_cache()

    print(f"Pseudo-labeling: promoted {total_promoted} residues to positive "
          f"(threshold={threshold})")
    return dataset.remove_columns('labels').add_column('labels', new_labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.label_smoothing and args.nnpu:
        print("Warning: --label-smoothing and --nnpu are mutually redundant. "
              "nnPU already treats negatives as unlabeled; --label-smoothing will be ignored.")
        args.label_smoothing = False

    print(f"Using device: {device}")
    print(f"Config: epochs={args.epochs}, lr={args.lr}, output={args.output}")
    print(f"  label_smoothing : {args.label_smoothing}"
          + (f" (alpha={args.label_smoothing_alpha})" if args.label_smoothing else ""))
    print(f"  nnpu            : {args.nnpu}"
          + (f" (prior={args.nnpu_prior or 'estimated'})" if args.nnpu else ""))
    print(f"  pseudo_labels   : {args.pseudo_labels}"
          + (f" (threshold={args.pseudo_label_threshold}, "
             f"epoch={args.pseudo_label_epoch})" if args.pseudo_labels else ""))

    finetuned_model = finetuning_utils.FinetunedEsmModel(MODEL_NAME).half().to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = finetuning_utils.process_sequence_dataset(F'{DATA_DIRECTORY}/data-extraction/scPDB_enhanced_binding_sites_translated_filtered.csv', tokenizer)
    # train_dataset = finetuning_utils.process_sequence_dataset(F'{DATA_DIRECTORY}/data-extraction/full_scPDB_translated_filtered.csv', tokenizer)
    val_dataset = finetuning_utils.process_sequence_dataset(
        f'{DATA_DIRECTORY}/data-extraction/ligysis_without_unobserved.csv',
        tokenizer)

    partial_collate_fn = functools.partial(finetuning_utils.collate_fn, tokenizer=tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                                  collate_fn=partial_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=int(val_dataset.num_rows / 20),
                                collate_fn=partial_collate_fn)

    optimizer = bnb.optim.AdamW8bit(finetuned_model.parameters(), lr=args.lr, eps=1e-4)

    # Compute class weights from the last training batch (preserves original behaviour)
    for batch in train_dataloader:
        labels = batch['labels']
    class_labels = labels.cpu().numpy().reshape(-1)[labels.cpu().numpy().reshape(-1) >= 0]
    weights = baseline_utils.compute_class_weights(class_labels)
    class_weights = torch.tensor(weights, device=device)
    del labels, class_labels

    # Standard BCE loss — also used for validation regardless of training-loss mode
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])

    # Estimate class prior for nnPU
    prior = None
    if args.nnpu:
        if args.nnpu_prior is not None:
            prior = args.nnpu_prior
        else:
            all_labels = np.concatenate([np.array(row) for row in train_dataset['labels']])
            prior = float((all_labels == 1).mean())
        print(f"nnPU prior π = {prior:.4f}")

    # Freeze LLM layers for the initial epochs
    for name, param in finetuned_model.named_parameters():
        if name.startswith('llm'):
            param.requires_grad = False

    test_losses = []
    train_losses = []

    for epoch in range(args.epochs):

        # Unfreeze LLM from epoch 2 onwards
        if epoch > 1:
            for name, param in finetuned_model.named_parameters():
                param.requires_grad = True

        # Pseudo-labeling pass — run once, just before the target epoch
        if args.pseudo_labels and epoch == args.pseudo_label_epoch:
            print(f"Running pseudo-labeling pass before epoch {epoch} ...")
            train_dataset = run_pseudo_labeling(
                finetuned_model, train_dataset, tokenizer,
                threshold=args.pseudo_label_threshold, device=device,
            )
            train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                                          collate_fn=partial_collate_fn)

        # ------------------------------------------------------------------
        # VALIDATION
        # ------------------------------------------------------------------
        finetuned_model.eval()
        with torch.no_grad():
            logits_list = []
            labels_list = []

            for batch in val_dataloader:
                output = finetuned_model(batch)
                labels = batch['labels'].to(device)
                flattened_labels = labels.flatten()

                cbs_logits = output.flatten()[flattened_labels != -100]
                valid_flattened_labels = labels.flatten()[flattened_labels != -100]

                logits_list.append(cbs_logits.cpu().float().detach().numpy())
                labels_list.append(valid_flattened_labels.cpu().float().detach().numpy())

                del labels, cbs_logits, valid_flattened_labels, flattened_labels
                gc.collect()
                torch.cuda.empty_cache()

            cbs_logits = torch.tensor(np.concatenate(logits_list)).to(device)
            valid_flattened_labels = torch.tensor(np.concatenate(labels_list)).to(device)

            predictions = torch.round(torch.sigmoid(cbs_logits))
            test_loss = loss_fn(cbs_logits, valid_flattened_labels)
            test_losses.append(test_loss.cpu().float().detach().numpy())

            test_acc = baseline_utils.accuracy_fn(y_true=valid_flattened_labels,
                                                  y_pred=predictions)
            fpr, tpr, _ = metrics.roc_curve(
                valid_flattened_labels.cpu().float().numpy(),
                torch.sigmoid(cbs_logits).cpu().float().numpy())
            roc_auc = metrics.auc(fpr, tpr)
            mcc = metrics.matthews_corrcoef(valid_flattened_labels.cpu().float().numpy(),
                                            predictions.cpu().float().numpy())
            f1 = metrics.f1_score(valid_flattened_labels.cpu().float().numpy(),
                                  predictions.cpu().float().numpy(), average='weighted')
            precision, recall, _ = metrics.precision_recall_curve(
                valid_flattened_labels.cpu().float().numpy(),
                torch.sigmoid(cbs_logits).cpu().float().numpy())
            auprc = metrics.auc(recall, precision)

        # ------------------------------------------------------------------
        # TRAINING
        # ------------------------------------------------------------------
        finetuned_model.train()
        batch_losses = []

        for batch in train_dataloader:
            output = finetuned_model(batch)
            labels = batch['labels'].to(device)
            flattened_labels = labels.flatten()

            cbs_logits = output.flatten()[flattened_labels != -100]
            valid_flattened_labels = labels.flatten()[flattened_labels != -100]

            loss = compute_train_loss(cbs_logits, valid_flattened_labels, args, loss_fn, prior)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.cpu().float().detach().numpy())

            del labels, output, cbs_logits, valid_flattened_labels, flattened_labels
            gc.collect()
            torch.cuda.empty_cache()

        train_losses.append(sum(batch_losses) / len(batch_losses))
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {test_acc:.2f}% | "
              f"Test loss: {test_loss:.5f}, AUC: {roc_auc:.4f}, MCC: {mcc:.4f}, "
              f"F1: {f1:.4f}, AUPRC: {auprc:.4f}, "
              f"sum: {sum(predictions.to(dtype=torch.int))}")

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------
    finetuned_model.eval()
    with torch.no_grad():
        logits_list = []
        labels_list = []

        for batch in val_dataloader:
            output = finetuned_model(batch)
            labels = batch['labels'].to(device)
            flattened_labels = labels.flatten()

            cbs_logits = output.flatten()[flattened_labels != -100]
            valid_flattened_labels = labels.flatten()[flattened_labels != -100]

            logits_list.append(cbs_logits.cpu().float().detach().numpy())
            labels_list.append(valid_flattened_labels.cpu().float().detach().numpy())

            del labels, cbs_logits, valid_flattened_labels, flattened_labels
            gc.collect()
            torch.cuda.empty_cache()

        cbs_logits = torch.tensor(np.concatenate(logits_list)).to(device)
        valid_flattened_labels = torch.tensor(np.concatenate(labels_list)).to(device)

        predictions = torch.round(torch.sigmoid(cbs_logits))
        test_loss = loss_fn(cbs_logits, valid_flattened_labels)
        test_losses.append(test_loss.cpu().float().detach().numpy())

        test_acc = baseline_utils.accuracy_fn(y_true=valid_flattened_labels,
                                                y_pred=predictions)
        fpr, tpr, _ = metrics.roc_curve(
            valid_flattened_labels.cpu().float().numpy(),
            torch.sigmoid(cbs_logits).cpu().float().numpy())
        roc_auc = metrics.auc(fpr, tpr)
        mcc = metrics.matthews_corrcoef(valid_flattened_labels.cpu().float().numpy(),
                                        predictions.cpu().float().numpy())
        f1 = metrics.f1_score(valid_flattened_labels.cpu().float().numpy(),
                                predictions.cpu().float().numpy(), average='weighted')
        precision, recall, _ = metrics.precision_recall_curve(
            valid_flattened_labels.cpu().float().numpy(),
            torch.sigmoid(cbs_logits).cpu().float().numpy())
        auprc = metrics.auc(recall, precision)
    
    print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {test_acc:.2f}% | "
              f"Test loss: {test_loss:.5f}, AUC: {roc_auc:.4f}, MCC: {mcc:.4f}, "
              f"F1: {f1:.4f}, AUPRC: {auprc:.4f}, "
              f"sum: {sum(predictions.to(dtype=torch.int))}")


    torch.save(finetuned_model, args.output)
    print(f"Model saved to {args.output}")


if __name__ == '__main__':
    main()
