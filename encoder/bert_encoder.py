import logging
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig
import json, os

class BERTEncoder(nn.Module):
    def __init__(self, pretrain_path):

        super().__init__()
        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = BertModel.from_pretrained(pretrain_path)
    def forward(self, token, att_mask):
        x = self.bert(token, attention_mask=att_mask)
        return x[0]