import torch
import sklearn.metrics
from torch import nn, optim
from torch.autograd import Variable
from .dataloaders import SentenceRELoader
from tqdm import tqdm
import os

def eval_semeval_result(config, data, id2rel):
    with open(config.semeval_answer, 'w') as file:
        for i, label in enumerate(data):
            relation = id2rel[label]
            format_result = '{0}	{1}'.format(8001 + i, relation)
            file.write(format_result)
            file.write('\n')
    state_code = os.system(config.eval_script)
    with open(config.semeval_result) as result:
        data = result.readlines()[-1][-11:-6]
    return data


class AverageMeter(object):
    """
    Computes and stores the average and current value of metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """
        String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class Sentence_RE(nn.Module):

    def __init__(self,
                 model,
                 train_path,
                 val_path,
                 test_path,
                 rel2id,
                 pretrain_path,
                 ckpt,
                 batch_size=16,
                 max_epoch=100,
                 lr=0.1,
                 weight_decay=1e-5,
                 num_workers=8):

        super().__init__()
        self.max_epoch = max_epoch
        # Load data
        self.train_singla_loader = SentenceRELoader(
            train_path,
            rel2id,
            pretrain_path,
            batch_size,
            True,
            num_workers=num_workers)
        self.val_loader = SentenceRELoader(
            val_path,
            rel2id,
            pretrain_path,
            batch_size,
            False,
            num_workers=num_workers)
        if test_path != None:
            self.test_loader = SentenceRELoader(
                test_path,
                rel2id,
                pretrain_path,
                batch_size,
                False,
                num_workers=num_workers)
        self.model = model
        # Criterion
        self.loss_func = nn.BCELoss()
        # Params and optimizer
        params = self.model.parameters()
        self.lr = lr
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, params), lr, weight_decay=weight_decay)
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt

    def train_semeval_model(self, warmup=True):
        best_f1 = 0
        item = 0
        global_step = 0
        for epoch in range(self.max_epoch):
            # Train
            print("=== Epoch %d train ===" % epoch)
            self.train_once(self.train_singla_loader, warmup)
            print("=== Epoch %d val ===" % epoch)
            result = self.eval_semeval(self.val_loader)
            print("acc: %.4f" % result['acc'])
            print("macro_f1: %.4f" % (result['macro_f1']))
            print("micro_f1: %.4f" % (result['micro_f1']))
            
            #set training criteria in config, micro_f1 or macro_f1
            if result[self.model.config.training_criteria] > best_f1:
                print("Best ckpt and saved.")
                torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
                best_f1 = result[self.model.config.training_criteria]
                item=0 #reset the early stopping counter
            else:
                item += 1
                #set early stopping patience level in config
                if item > self.model.config.patience:
                    print('Epoch %05d: early stopping' % (epoch + 1))
                    break
        print("Best f1 on val set: %f" % (best_f1))

    def train_once(self, loader, warmup=True):
        self.model.train()
        global_step = 0
        avg_sent_loss = AverageMeter()
        avg_sent_acc = AverageMeter()
        t = tqdm(loader)
        for iter, data in enumerate(t):
            if torch.cuda.is_available():
                for i in range(len(data)):
                    try:
                        data[i] = data[i].cuda()
                    except:
                        pass
            # sentence
            sent_label = data[0]
            args = data[1:]
            logits, _ = self.model(sent_label, *args)
            # loss
            loss, acc = self.acc_loss(logits, sent_label, 19)
            # Log
            avg_sent_loss.update(loss.item(), 1)
            avg_sent_acc.update(acc, 1)
            t.set_postfix(sent_loss=avg_sent_loss.avg, sent_acc=avg_sent_acc.avg)
            # Optimize
            if warmup == True:
                warmup_step = 300
                if global_step < warmup_step:
                    warmup_rate = float(global_step) / warmup_step
                else:
                    warmup_rate = 1.0
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr * warmup_rate
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            self.optimizer.zero_grad()
            global_step += 1

    def eval_semeval(self, eval_loader):
        self.model.eval()
        avg_acc = AverageMeter()
        pred_result = []
        label_tot = []
        with torch.no_grad():
            t = tqdm(eval_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                args = data[1:]

                logits, pred = self.model(label, *args)
                for i in range(pred.size(0)):
                    pred_result.append(pred[i].item())
                    label_tot.append(label[i].item())
                # Log
                acc = float((pred == label).long().sum()) / label.size(0)
                avg_acc.update(acc, pred.size(0))
                t.set_postfix(acc=avg_acc.avg)
            macro_f1 = sklearn.metrics.f1_score(label_tot, pred_result, average='macro')
            micro_f1 = sklearn.metrics.f1_score(label_tot, pred_result, average='micro')
            out = {'macro_f1': macro_f1, 'micro_f1': micro_f1, 'acc': avg_acc.avg}
            # office script
            if len(pred_result) > 2716:
                result = eval_semeval_result(self.model.config, pred_result, self.test_loader.dataset.id2rel)
                print('official script semeval test result: {}'.format(result))
            print(str(out))
        return out

    def acc_loss(self, logits, sent_label, nums=19):
        y = Variable(torch.eye(nums))
        if torch.cuda.is_available():
            y = y.cuda()
        y = y.index_select(dim=0, index=sent_label.data)
        _, pred = torch.max(logits.view(-1, nums), 1)
        acc = float((pred == sent_label).long().sum()) / sent_label.size(0)
        sent_loss = self.loss_func(logits, y)
        return sent_loss, acc

    def load_state_dict(self, ckpt):
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['state_dict'])
