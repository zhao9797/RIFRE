import torch
import torch.nn as nn
import torch.nn.functional as F
import json


class RIFRE_TR(nn.Module):

    def __init__(self, encoder, config):
        super(RIFRE_TR, self).__init__()
        self.config = config
        self.sentence_encoder = encoder
        self.gat = HGAT(config)


    def forward(self, token, mask, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails):

        # token
        hidden = self.sentence_encoder(token, mask)

        sub_heads_logits, sub_tails_logits, obj_heads_logits, obj_tails_logits = self.gat(hidden, sub_head, sub_tail, mask)


        #loss
        sub_head_loss = F.binary_cross_entropy(sub_heads_logits, sub_heads.float(), reduction='none')
        sub_head_loss = (sub_head_loss * mask.float()).sum() / mask.float().sum()
        sub_tail_loss = F.binary_cross_entropy(sub_tails_logits, sub_tails.float(), reduction='none')
        sub_tail_loss = (sub_tail_loss * mask.float()).sum() / mask.float().sum()
        obj_head_loss = F.binary_cross_entropy(obj_heads_logits, obj_heads.float(), reduction='none').sum(2)
        obj_head_loss = (obj_head_loss * mask.float()).sum() / mask.float().sum()
        obj_tail_loss = F.binary_cross_entropy(obj_tails_logits, obj_tails.float(), reduction='none').sum(2)
        obj_tail_loss = (obj_tail_loss * mask.float()).sum() / mask.float().sum()
        loss = sub_head_loss + sub_tail_loss + obj_head_loss + obj_tail_loss
        return loss
    def predict_sub(self, token, mask=None):
        t = self.sentence_encoder(token, mask)
        sub_heads_logits, sub_tails_logits = self.gat(t, mask=mask)
        return sub_heads_logits, sub_tails_logits

    def predict_obj(self, token, sub_head, sub_tail, mask=None):
        hidden = self.sentence_encoder(token, mask)
        _, _, obj_heads_logits, obj_tails_logits = self.gat(hidden, sub_head, sub_tail, mask)


        return obj_heads_logits, obj_tails_logits


class HGAT(nn.Module):
    def __init__(self, config):
        super(HGAT, self).__init__()
        self.config = config
        hidden_size = config.hidden_size
        self.embeding = nn.Embedding(config.class_nums, hidden_size)
        self.relation = nn.Linear(hidden_size, hidden_size)
        self.down = nn.Linear(3 * hidden_size, hidden_size)
        #t
        self.start_head = nn.Linear(hidden_size, 1)
        self.end_head = nn.Linear(hidden_size, 1)
        self.start_tail = nn.Linear(hidden_size, 1)
        self.end_tail = nn.Linear(hidden_size, 1)
        self.layers = nn.ModuleList([GATLayer(hidden_size) for _ in range(config.gat_layers)])

    def forward(self, x, sub_head =None, sub_tail =None, mask=None):
        # relation

        p = torch.arange(self.config.class_nums).long()
        if torch.cuda.is_available():
            p = p.cuda()
        p = self.relation(self.embeding(p))
        p = p.unsqueeze(0).expand(x.size(0), p.size(0), p.size(1))  # bcd
        x, p = self.gat_layer(x, p, mask)  # x bcd
        ts, te = self.pre_head(x)
        if sub_head is not None and sub_tail is not None:
            e1 = self.entity_trans(x, sub_head, sub_tail)
            hs, he = self.pre_tail(x, e1, p)
            return ts, te, hs, he
        return ts, te
    def extra_entity(self, hidden, sub_head, sub_tail):
        onehot_head = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_tail = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_head = onehot_head.scatter_(1, sub_head.view(-1, 1), 1)
        onehot_tail = onehot_tail.scatter_(1, sub_tail.view(-1, 1), 1)
        head_hidden = (onehot_head.unsqueeze(2) * hidden).sum(1)  # (B, H)
        tail_hidden = (onehot_tail.unsqueeze(2) * hidden).sum(1)  # (B, H)
        entity = (head_hidden + tail_hidden) / 2
        return entity
    def entity_trans(self, x, sub_head, sub_tail):
        batch = x.size(0)
        avai_len = x.size(1)
        sub_head, sub_tail = sub_head.view(-1, 1), sub_tail.view(-1, 1)
        mask = []
        for i in range(batch):
            pos = torch.zeros(avai_len, device=x.device).float()
            h, t = sub_head[i].item(), sub_tail[i].item()
            pos[h:t+1] = 1.0
            mask.append(pos)
        mask = torch.stack(mask, 0)
        e1 = x * mask.unsqueeze(2).expand(-1, -1, x.size(2))
        # avg
        if self.config.pool_type == 'avg':
            divied = torch.sum(mask, 1)
            e1 = torch.sum(e1, 1) / divied.unsqueeze(1)
        elif self.config.pool_type == 'max':
            # max
            e1, _ = torch.max(e1, 1)
        return e1
    def pre_head(self, x):
        x = torch.tanh(x)
        ts = self.start_head(x).squeeze(2)
        ts = ts.sigmoid()
        te = self.end_head(x).squeeze(2)
        te = te.sigmoid()
        return ts, te
    def pre_tail(self, x, e1, p):
        e1 = e1.unsqueeze(1).expand_as(x)
        e1 = e1.unsqueeze(2).expand(-1, -1, p.size(1), -1)
        x = x.unsqueeze(2).expand(-1,-1,p.size(1),-1)
        p = p.unsqueeze(1).expand(-1,x.size(1),-1,-1)
        t = self.down(torch.cat([x, p, e1], 3))
        t = torch.tanh(t)
        ts = self.start_tail(t).squeeze(3)
        ts = ts.sigmoid()
        te = self.end_tail(t).squeeze(3)
        te = te.sigmoid()
        return ts, te

    def gat_layer(self, x, p, mask=None):

        for m in self.layers:
            x, p = m(x, p, mask)
        return x, p


class GATLayer(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.ra1 = RelationAttention(hidden_size)
        self.ra2 = RelationAttention(hidden_size)

    def forward(self, x, p, mask=None):
        x_ = self.ra1(x, p)
        x = x_+ x
        p_ = self.ra2(p, x, mask)
        p =  p_ + p
        return x, p

class RelationAttention(nn.Module):
    def __init__(self, hidden_size):
        super(RelationAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(2 * hidden_size, 1)
        self.gate = nn.Linear(hidden_size * 2, 1)

    def forward(self, p, x, mask=None):
        q = self.query(p)
        k = self.key(x)
        score = self.fuse(q, k)
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