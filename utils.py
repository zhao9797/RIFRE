import logging
def count_params(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    print('total number of parameters: %d\n\n' % param_count)


def eval_f1(pred_result, label, rel2id):
    correct = 0
    total = len(label)
    correct_positive = 0
    pred_positive = 0
    gold_positive = 0
    neg = -1
    for name in ['NA', 'na', 'no_relation', 'Other', 'Others']:
        if name in rel2id:
            neg = rel2id[name]
            break
    for i in range(total):
        golden = label[i]
        if golden == pred_result[i]:
            correct += 1
            if golden != neg:
                correct_positive += 1
        if golden != neg:
            gold_positive += 1
        if pred_result[i] != neg:
            pred_positive += 1
    acc = float(correct) / float(total)
    try:
        micro_p = float(correct_positive) / float(pred_positive)
    except:
        micro_p = 0
    try:
        micro_r = float(correct_positive) / float(gold_positive)
    except:
        micro_r = 0
    try:
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
    except:
        micro_f1 = 0
    result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}
    print('Evaluation result: {}.'.format(result))
    return result