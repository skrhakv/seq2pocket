### COPIED FROM https://github.com/skrhakv/cryptic-finetuning/blob/master/src/finetuning_utils.py

from transformers import EsmModel
import torch
import torch.nn as nn
import numpy as np
import warnings

warnings.filterwarnings('ignore')
torch.manual_seed(42)

DROPOUT = 0.3
OUTPUT_SIZE = 1
MAX_LENGTH = 1024
LABEL_PAD_TOKEN_ID = -100

class FinetunedEsmModel(nn.Module):
    def __init__(self, esm_model: str) -> None:
        super().__init__()
        self.llm = EsmModel.from_pretrained(esm_model)

        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(self.llm.config.hidden_size, OUTPUT_SIZE)
        
    def forward(self, batch: dict[str, np.ndarray]) -> torch.Tensor:
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        token_embeddings = self.llm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        return self.classifier(token_embeddings)
