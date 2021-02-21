import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RIFRE_SEN(nn.Module):

    def __init__(self, encoder, config):
        super(RIFRE_SEN, self).__init__()
        self.config = config
        self.sentence_encoder = encoder
        self.gat = HGAT(config)


    def forward(self, label, token, pos1, pos2, mask=None):
        # Check size of tensors
        x = self.sentence_encoder(token, mask)
        logits = self.gat(x, pos1, pos2, mask)
        _, pred = torch.max(logits.view(-1, self.config.class_nums), 1)
        return logits, pred


class HGAT(nn.Module):
    def __init__(self, config):
        super(HGAT, self).__init__()
        self.config = config
        hidden_size = config.hidden_size
        self.embeding = nn.Embedding(config.class_nums, hidden_size)
        self.relation = nn.Linear(hidden_size, hidden_size)
        self.fc1 = nn.Linear(3*hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.layers = nn.ModuleList([GATLayer(hidden_size) for _ in range(config.gat_layers)])

    def forward(self, x, pos1, pos2, mask=None, pretrian=False):
        p = torch.arange(self.config.class_nums, device = x.device).long()
        p = self.embeding(p)
        p = self.relation(p)
        p = p.unsqueeze(0).expand(x.size(0), p.size(0), p.size(1))  # bcd
        x, p = self.gat_layer(x, p, mask)
        e1 = self.entity_trans(x, pos1)
        e2 = self.entity_trans(x, pos2)

        p = torch.cat([p, e1.unsqueeze(1).expand_as(p), e2.unsqueeze(1).expand_as(p)], 2)
        p = self.fc1(p)
        p = torch.tanh(p)
        p = self.fc2(p).squeeze(2).sigmoid()
        return p

    def gat_layer(self, x, p, mask=None):
        for m in self.layers:
            x, p = m(x, p, mask)
        return x, p

    def entity_trans(self, x, pos):
        e1 = x * pos.unsqueeze(2).expand(-1, -1, x.size(2))
        # avg
        if self.config.pool_type == 'avg':
            divied = torch.sum(pos, 1)
            e1 = torch.sum(e1, 1) / divied.unsqueeze(1)
        elif self.config.pool_type == 'max':
            # max
            e1, _ = torch.max(e1, 1)
        return e1


class GATLayer(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.ra1 = RelationAttention(hidden_size)
        self.ra2 = RelationAttention(hidden_size)

    def forward(self, x, p, mask=None):
        x = self.ra1(x, p) + x
        p = self.ra2(p, x, mask) + p
        return x, p

class RelationAttention(nn.Module):
    def __init__(self, hidden_size):
        super(RelationAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(2*hidden_size, 1)
        self.gate = nn.Linear(hidden_size * 2, 1)

    def forward(self, p, x, mask=None):
        q = self.query(p)#bcd
        k = self.key(x)#bld
        score = self.fuse(q, k)#bcl
        if mask is not None:
            mask = 1 - mask[:, None, :].expand(-1, score.size(1), -1)
            score = score.masked_fill(mask == 1, -1e9)
        score = F.softmax(score, 2)
        v = self.value(x)
        out = torch.einsum('bcl,bld->bcd', score, v) + p
        g = self.gate(torch.cat([out, p], 2)).sigmoid()
        out = g * out + (1 - g) * p
        return out

    def fuse(self, x, y):
        x = x.unsqueeze(2).expand(-1, -1, y.size(1), -1)
        y = y.unsqueeze(1).expand(-1, x.size(1), -1, -1)
        temp = torch.cat([x, y], 3)
        return self.score(temp).squeeze(3)