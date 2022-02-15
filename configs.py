import os
import json
import random


class Config(object):
    def __init__(self):
        root_path = 'datasets'
        self._get_path(root_path)
        self.batch_size = 6
        self.max_length = 100
        self.epoch = 50
        self.lr = 1e-1
        
        self.patience = 9 #early stopping patience level
        self.training_criteria = 'micro_f1' #or 'macro_f1'

        self.gat_layers = 2
        self.hidden_size = 768

        self.nyt_class = 24
        self.semeval_class = 19
        self.webnlg_class = 171
        self.fewrel_class = 80
        self.class_nums = None

        self.seed = 2020

        self.pool_type = 'avg'
        if not os.path.exists('checkpoint'):
            os.mkdir('checkpoint')
        self.semeval_ckpt = 'checkpoint/semeval.pth.tar'
        self.webnlg_ckpt = 'checkpoint/webnlg.pth.tar'
        self.nyt_ckpt = 'checkpoint/nyt.pth.tar'

        semeval_eval = './eval'
        self.semeval_answer = os.path.join(semeval_eval, 'SemEval2010_task8_scorer-v1.2/result.txt')
        self.semeval_keys = os.path.join(semeval_eval, 'SemEval2010_task8_scorer-v1.2/answer_keys.txt')
        self.semeval_script = os.path.join(semeval_eval,
                                           'SemEval2010_task8_scorer-v1.2/semeval2010_task8_scorer-v1.2.pl')
        self.semeval_result = os.path.join(semeval_eval, 'SemEval2010_task8_scorer-v1.2/semeval_score.txt')
        self.eval_script = "perl {0} {1} {2} > {3}".format(self.semeval_script, self.semeval_answer, self.semeval_keys,
                                                           self.semeval_result)

    def _get_path(self, root_path):
        self.root_path = root_path
        # bert base uncase bert\bert-base-uncased
        self.bert_base = os.path.join(root_path, 'bert/bert-base-uncased')
        self.bert_base_case = os.path.join(root_path, 'bert/bert-base-cased')

        # semeval
        self.semeval_rel2id = os.path.join(root_path, 'data/new_semeval/srel2id.json')
        self.semeval_train = os.path.join(root_path, 'data/new_semeval/train.txt')
        self.semeval_val = os.path.join(root_path, 'data/new_semeval/dev.txt')
        self.semeval_test = os.path.join(root_path, 'data/new_semeval/test.txt')

        # webnlg-triple
        self.webnlg_rel2id = os.path.join(root_path, 'data/webnlg/rel2id.json')
        self.webnlg_train = os.path.join(root_path, 'data/webnlg/train_triples.json')
        self.webnlg_val = os.path.join(root_path, 'data/webnlg/dev_triples.json')
        self.webnlg_test = os.path.join(root_path, 'data/webnlg/test_triples.json')
        self.webnlg_bynum = [os.path.join(root_path, 'data/webnlg/test_split_by_num/test_triples_{}.json'.format(i + 1))
                             for i in range(5)]
        self.webnlg_bytype = [os.path.join(root_path, 'data/webnlg/test_split_by_type/test_triples_{}.json'.format(x))
                              for x in ['normal', 'seo', 'epo']]
        # nyt-triple
        self.nyt_rel2id = os.path.join(root_path, 'data/nyt/rel2id.json')
        self.nyt_train = os.path.join(root_path, 'data/nyt/train_triples.json')
        self.nyt_val = os.path.join(root_path, 'data/nyt/dev_triples.json')
        self.nyt_test = os.path.join(root_path, 'data/nyt/test_triples.json')
        self.nyt_bynum = [os.path.join(root_path, 'data/nyt/test_split_by_num/test_triples_{}.json'.format(i + 1)) for i
                          in range(5)]
        self.nyt_bytype = [os.path.join(root_path, 'data/nyt/test_split_by_type/test_triples_{}.json'.format(x)) for x
                           in ['normal', 'seo', 'epo']]
