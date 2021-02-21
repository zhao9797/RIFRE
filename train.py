from encoder.bert_encoder import BERTEncoder
from models.rifre_sentence import RIFRE_SEN
from models.rifre_triple import RIFRE_TR
from framework.sentence_re import Sentence_RE
from framework.triple_re import Triple_RE
from configs import Config
from utils import count_params
import numpy as np
import torch
import random, argparse
torch.cuda.set_device(0)

def seed_torch(seed=2020):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Controller')
    parser.add_argument('--train', default=True, type=bool)
    parser.add_argument('--dataset', default='webnlg', type=str,
                        help='specify the dataset from ["nyt","webnlg","semeval"]')
    args = parser.parse_args()
    dataset = args.dataset
    is_train = args.train
    config = Config()
    if config.seed is not None:
        print(config.seed)
        seed_torch(config.seed)

    if dataset == 'semeval':
        print('train--' + dataset)
        config.class_nums = config.semeval_class
        sentence_encoder = BERTEncoder(pretrain_path=config.bert_base)
        model = RIFRE_SEN(sentence_encoder, config)
        count_params(model)
        framework = Sentence_RE(model,
                                train_path=config.semeval_train,
                                val_path=config.semeval_val,
                                test_path=config.semeval_test,
                                rel2id=config.semeval_rel2id,
                                pretrain_path=config.bert_base,
                                ckpt=config.semeval_ckpt,
                                batch_size=config.batch_size,
                                max_epoch=config.epoch,
                                lr=config.lr)
        framework.train_semeval_model()
        framework.load_state_dict(config.semeval_ckpt)
        print('test:')
        framework.eval_semeval(framework.test_loader)
    elif dataset == 'webnlg':
        print('train--' + dataset + config.webnlg_ckpt)
        config.class_nums = config.webnlg_class
        sentence_encoder = BERTEncoder(pretrain_path=config.bert_base_case)

        model = RIFRE_TR(sentence_encoder, config)
        count_params(model)
        framework = Triple_RE(model,
                              train=config.webnlg_train,
                              val=config.webnlg_val,
                              test=config.webnlg_test,
                              rel2id=config.webnlg_rel2id,
                              pretrain_path=config.bert_base_case,
                              ckpt=config.webnlg_ckpt,
                              batch_size=config.batch_size,
                              max_epoch=config.epoch,
                              lr=config.lr,
                              num_workers=4)

        framework.train_model()
        framework.load_state_dict(config.webnlg_ckpt)
        print('test:' + config.webnlg_ckpt)
        framework.test_set.metric(framework.model)
    elif dataset == 'nyt':
        print('train--' + dataset)
        config.class_nums = config.nyt_class
        sentence_encoder = BERTEncoder(pretrain_path=config.bert_base_case)
        model = RIFRE_TR(sentence_encoder, config)
        count_params(model)
        framework = Triple_RE(model,
                              train=config.nyt_train,
                              val=config.nyt_val,
                              test=config.nyt_test,
                              rel2id=config.nyt_rel2id,
                              pretrain_path=config.bert_base_case,
                              ckpt=config.nyt_ckpt,
                              batch_size=config.batch_size,
                              max_epoch=config.epoch,
                              lr=config.lr)
        output_path = 'save_result/nyt_result.json'
        framework.train_model()
        framework.load_state_dict(config.nyt_ckpt)
        print('test:' + config.nyt_ckpt)
        framework.test_set.metric(framework.model, output_path=output_path)
    else:
        print('unkonw dataset')