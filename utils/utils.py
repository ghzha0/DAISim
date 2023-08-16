import os
import random

import numpy as np
import torch
from torch import Tensor
import json
import pandas as pd
from data import ExpertDataSet, EarlyLateExpertDataset


def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))


def tensor2list(tensor: Tensor):
    return tensor.cpu().tolist()


def length2mask(length, max_len, valid_mask_val, invalid_mask_val):
    mask = []

    if isinstance(valid_mask_val, Tensor):
        valid_mask_val = tensor2list(valid_mask_val)
    if isinstance(invalid_mask_val, Tensor):
        invalid_mask_val = tensor2list(invalid_mask_val)
    if isinstance(length, Tensor):
        length = tensor2list(length)

    for _len in length:
        mask.append([valid_mask_val] * _len + [invalid_mask_val] * (max_len - _len))

    return torch.tensor(mask)


def get_sequence_mask(shape, sequence_length, axis=1):
    assert axis <= len(shape)
    mask_shape = shape[axis + 1:]

    valid_mask_val = torch.ones(mask_shape)
    invalid_mask_val = torch.zeros(mask_shape)

    max_len = shape[axis]

    return length2mask(sequence_length, max_len, valid_mask_val, invalid_mask_val)


def sequence_mask(tensor: Tensor, sequence_length, axis=1):
    mask = get_sequence_mask(tensor.shape, sequence_length, axis).to(tensor.device)
    return tensor * mask


def load_arg_train_test(content_path):
    question_tag = pd.read_csv(content_path, usecols=['question_id', 'tags'])
    ids = list(question_tag['question_id'])
    tags = list(question_tag['tags'])
    exer_tags = {ids[i]: tags[i] for i in range(len(ids))}

    with open(os.path.join('baseline/stugail/train_data.json')) as F:
        train_data = json.load(F)

    with open(os.path.join('baseline/stugail/arg_data.json')) as F:
        arg_data = json.load(F)

    with open(os.path.join('baseline/stugail/test_data.json')) as F:
        test_data = json.load(F)

    return EarlyLateExpertDataset(train_data, train_data, exer_tags), \
           EarlyLateExpertDataset(arg_data, arg_data, exer_tags), \
           EarlyLateExpertDataset(test_data, test_data, exer_tags)


def load_train_test(data_path, content_path, length):
    question_tag = pd.read_csv(content_path, usecols=['question_id', 'tags'])
    ids = list(question_tag['question_id'])
    tags = list(question_tag['tags'])
    exer_tags = {ids[i]: tags[i] for i in range(len(ids))}
    if length == '20':
        with open(os.path.join(data_path, 'train_early_log.json')) as F:
            train_early_data = json.load(F)

        with open(os.path.join(data_path, 'train_late_log.json')) as F:
            train_late_data = json.load(F)

        with open(os.path.join(data_path, 'test_early_log.json')) as F:
            test_early_data = json.load(F)

        with open(os.path.join(data_path, 'test_late_log.json')) as F:
            test_late_data = json.load(F)
    else:
        with open(os.path.join(data_path, f'train_early_log_{length}.json')) as F:
            train_early_data = json.load(F)

        with open(os.path.join(data_path, f'train_late_log_{length}.json')) as F:
            train_late_data = json.load(F)

        with open(os.path.join(data_path, f'test_early_log_{length}.json')) as F:
            test_early_data = json.load(F)

        with open(os.path.join(data_path, f'test_late_log_{length}.json')) as F:
            test_late_data = json.load(F)

    train_len = len(train_early_data) * 0.8
    return EarlyLateExpertDataset(train_early_data[:int(train_len)], train_late_data[:int(train_len)], exer_tags), \
           EarlyLateExpertDataset(train_early_data[int(train_len):], train_late_data[int(train_len):], exer_tags), \
           EarlyLateExpertDataset(test_early_data, test_late_data, exer_tags)


def split_test_set(test_set):
    early_data = test_set.early_data
    late_data = test_set.late_data
    exer_tags = test_set.exer_tag

    ans_list = []
    for data in early_data:
        ans_list.append(np.mean(data['answer']))
    index = np.argsort(ans_list)
    train_len = int(len(index) * 0.9)
    train_idx = index[:train_len]
    test_idx = index[train_len:]

    train_early_data = [early_data[i] for i in train_idx]
    train_late_data = [late_data[i] for i in train_idx]
    test_early_data = [early_data[i] for i in test_idx]
    test_late_data = [late_data[i] for i in test_idx]

    return EarlyLateExpertDataset(train_early_data, train_late_data, exer_tags), \
           EarlyLateExpertDataset(test_early_data, test_late_data, exer_tags)


def load_train_val_test(data_path, content_path):
    question_tag = pd.read_csv(content_path, usecols=['question_id', 'tags'])
    ids = list(question_tag['question_id'])
    tags = list(question_tag['tags'])
    exer_tags = {ids[i]: tags[i] for i in range(len(ids))}

    with open(os.path.join(data_path, 'train_early_log.json')) as F:
        train_early_data = json.load(F)

    with open(os.path.join(data_path, 'train_late_log.json')) as F:
        train_late_data = json.load(F)

    with open(os.path.join(data_path, 'test_early_log.json')) as F:
        test_early_data = json.load(F)

    with open(os.path.join(data_path, 'test_late_log.json')) as F:
        test_late_data = json.load(F)

    val_len = len(test_early_data) // 5

    return EarlyLateExpertDataset(train_early_data, train_late_data, exer_tags), \
           EarlyLateExpertDataset(test_early_data[:val_len], test_late_data[:val_len], exer_tags), \
           EarlyLateExpertDataset(test_early_data[val_len:], test_late_data[val_len:], exer_tags)


def get_str_for_eval(questions, answers):
    res = []
    for seq_q, seq_a in zip(questions, answers):
        temp = []
        for q, a in zip(seq_q, seq_a):
            if a == 1:
                temp.append(str(q))
            else:
                temp.append(str(q) + "w")
        res.append(" ".join(temp))
    return res
