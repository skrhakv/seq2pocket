import torch
import numpy as np
import sys 
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import functools
from sklearn import metrics
import gc
import bitsandbytes as bnb
from torch.utils.data import DataLoader
import warnings

sys.path.append('/home/skrhakv/cryptic-nn/src')
import baseline_utils
import finetuning_utils # download from https://github.com/skrhakv/cryptic-finetuning/blob/master/src/finetuning_utils.py


torch.manual_seed(0)

warnings.filterwarnings('ignore')
torch.manual_seed(42)

MODEL_NAME = 'facebook/esm2_t36_3B_UR50D'
DATASET = 'cryptobench'
DATA_PATH = f'/home/skrhakv/cryptic-nn/data/{DATASET}'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

finetuned_model = finetuning_utils.FinetunedEsmModel(MODEL_NAME).half().to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = finetuning_utils.process_sequence_dataset('/home/skrhakv/cryptoshow-analysis/data/E-regular-binding-site-predictor/scPDB_enhanced_binding_sites_translated.csv', tokenizer)
val_dataset = finetuning_utils.process_sequence_dataset('/home/skrhakv/cryptoshow-analysis/data/E-regular-binding-site-predictor/ligysis_without_unobserved.csv', tokenizer) 

partial_collate_fn = functools.partial(finetuning_utils.collate_fn, tokenizer=tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=partial_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=int(val_dataset.num_rows / 20), collate_fn=partial_collate_fn)

optimizer = bnb.optim.AdamW8bit(finetuned_model.parameters(), lr=0.0001, eps=1e-4) 

EPOCHS = 3

# precomputed class weights
for batch in train_dataloader:
    labels = batch['labels']

class_labels = labels.cpu().numpy().reshape(-1)[labels.cpu().numpy().reshape(-1) >= 0]
weights = baseline_utils.compute_class_weights(class_labels)
class_weights = torch.tensor(weights, device=device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])

del labels, class_labels

for name, param in finetuned_model.named_parameters():
     if name.startswith('llm'): 
        param.requires_grad = False

test_losses = []
train_losses = []

for epoch in range(EPOCHS):
    if epoch > 1:
        for name, param in finetuned_model.named_parameters():
            param.requires_grad = True

    finetuned_model.eval()

    # VALIDATION LOOP
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

        # TODO: is it going to fail on memory or not when using LYGISYS?
        cbs_logits = torch.tensor(np.concatenate(logits_list)).to(device)
        valid_flattened_labels = torch.tensor(np.concatenate(labels_list)).to(device)

        predictions = torch.round(torch.sigmoid(cbs_logits)) # (probabilities>0.95).float() # torch.round(torch.sigmoid(valid_flattened_cbs_logits))

        cbs_test_loss =  loss_fn(cbs_logits, valid_flattened_labels)

        test_loss = cbs_test_loss

        test_losses.append(test_loss.cpu().float().detach().numpy())

        # compute metrics on test dataset
        test_acc = baseline_utils.accuracy_fn(y_true=valid_flattened_labels,
                                y_pred=predictions)
        fpr, tpr, thresholds = metrics.roc_curve(valid_flattened_labels.cpu().float().numpy(), torch.sigmoid(cbs_logits).cpu().float().numpy())
        roc_auc = metrics.auc(fpr, tpr)

        mcc = metrics.matthews_corrcoef(valid_flattened_labels.cpu().float().numpy(), predictions.cpu().float().numpy())

        f1 = metrics.f1_score(valid_flattened_labels.cpu().float().numpy(), predictions.cpu().float().numpy(), average='weighted')

        precision, recall, thresholds = metrics.precision_recall_curve(valid_flattened_labels.cpu().float().numpy(), torch.sigmoid(cbs_logits).cpu().float().numpy())
        auprc = metrics.auc(recall, precision)

    
    finetuned_model.train()

    batch_losses = []

    # TRAIN
    for batch in train_dataloader:

        output = finetuned_model(batch)
        labels = batch['labels'].to(device)

        flattened_labels = labels.flatten()

        cbs_logits = output.flatten()[flattened_labels != -100]
        valid_flattened_labels = labels.flatten()[flattened_labels != -100]

        loss =  loss_fn(cbs_logits, valid_flattened_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.cpu().float().detach().numpy())
        
        del labels, output, cbs_logits, valid_flattened_labels, flattened_labels
        gc.collect()
        torch.cuda.empty_cache()

    train_losses.append(sum(batch_losses) / len(batch_losses))
    print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {test_acc:.2f}% | Test loss: {test_loss:.5f}, AUC: {roc_auc:.4f}, MCC: {mcc:.4f}, F1: {f1:.4f}, AUPRC: {auprc:.4f}, sum: {sum(predictions.to(dtype=torch.int))}")

OUTPUT_PATH = '/home/skrhakv/cryptoshow-analysis/data/E-regular-binding-site-predictor/model-enhanced-scPDB.pt'
torch.save(finetuned_model, OUTPUT_PATH)