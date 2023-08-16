import argparse
import logging
import os
import numpy as np
import torch
import random
from model import *
# 定义超参数
from utils import load_train_val_test

def parse_arg():
    parser = argparse.ArgumentParser('DAISim')
    parser.add_argument('--case', action='store_true', default=False)
    parser.add_argument('--no_gail', action='store_true', default=False)
    parser.add_argument('--no_di', action='store_true', default=False)
    parser.add_argument('--d_sample', action='store_true', default=False)
    parser.add_argument('--g_mean', action='store_true', default=False)

    parser.add_argument('--use_pre', action='store_true', default=False, help='use pretrain model parameters')
    parser.add_argument('--train', action='store_true', default=False, help='train model')
    parser.add_argument('--generate_syn_set', action='store_true', default=False, help='generate_syn_set')
    parser.add_argument('--generate_argumentation_set', action='store_true', default=False,
                        help='generate_argumentation_set')
    parser.add_argument('--cuda', default='0', help="use which gpu")
    parser.add_argument('--generator', default='mail', help="generator model")
    parser.add_argument('--kc_num', type=int, default=123, help="knowledge num")
    parser.add_argument('--exercise_num', type=int, default=17751, help="total exercise num")
    parser.add_argument('--user_num', type=int, default=4163, help="total user num")
    parser.add_argument('--length', default='20', help="generate length")

    train_settings = parser.add_argument_group('train_settings')
    train_settings.add_argument('--optim', default='adam', help='optimizer type')
    train_settings.add_argument('--pre_g_lr', type=float, default='0.001', help='learning rate of G in pretrain')
    train_settings.add_argument('--pre_d_lr', type=float, default='0.001', help='learning rate of D in pretrain')
    train_settings.add_argument('--pre_r_lr', type=float, default='0.001', help='learning rate of Rec in pretrain')
    train_settings.add_argument('--g_lr', type=float, default='0.0001', help='learning rate of G')
    train_settings.add_argument('--d_lr', type=float, default='0.0001', help='learning rate of D')
    train_settings.add_argument('--gailoss', type=float, default=0.1,
                                help='gailoss proportion')
    train_settings.add_argument('--alpha', type=float, default=0.5,
                                help='policy_surr')
    train_settings.add_argument('--tau', type=float, default=0.5,
                                help='pairwise and adversarial loss balance')
    train_settings.add_argument('--beta', type=float, default=0.5,
                                help='policy entropy')
    train_settings.add_argument('--gamma', type=float, default=0.1,
                                help='discount factor')
    train_settings.add_argument('--lam', type=float, default=0.1,
                                help='gae')
    train_settings.add_argument('--clip_epsilon', type=float, default=0.1,
                                help='ppo')
    train_settings.add_argument('--batch_size', type=int, default=256,
                                help='pretrain batch size')
    train_settings.add_argument('--generate_size', type=int, default=256,
                                help='train batch size')
    train_settings.add_argument('--ppo_epoch', type=int, default=4, help='steps train G')
    train_settings.add_argument('--epoch', type=int, default=1000, help='train epochs')
    train_settings.add_argument('--pre_epoch', type=int, default=10, help='pretrain epochs')
    train_settings.add_argument('--d_epoch', type=int, default=100, help='disc train epoch')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--embed_size', type=int, default=128, help='size of embedding')
    model_settings.add_argument('--hidden_size', type=int, default=128, help='size of gru hidden')
    model_settings.add_argument('--disc_hidden_size', type=int, default=128, help='size of disc gru hidden')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--log_name', default="log", help='name of log')
    path_settings.add_argument('--load_name', help='where to load model')
    path_settings.add_argument('--load_model_name', help='load which model')

    path_settings.add_argument('--data_path', default="data/assist0910/", help='data path')
    path_settings.add_argument('--tag_path', default="data/assist0910/exer_tag.csv", help='tag path')
    return parser.parse_args()


def train(args, train_set, val_set, test_set):
    """
    :param val_set:
    :param train_set:
    :param args:
    """
    mail = MAILModel(args)
    mail.train(args, train_set, val_set, test_set)

def run():
    """
    Run the whole system
    """
    args = parse_arg()
    train_set, valid_set, test_set = load_train_test(args.data_path, args.tag_path, args.length)

    if args.train:
        train(args, train_set, valid_set, test_set)

if __name__ == '__main__':
    run()
