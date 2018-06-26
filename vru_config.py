#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, BMW Group, all rights reserved.
#
# Redistribution and use in source and other forms, with or without modification,
# are permitted only in BMW internal usage. Any other companies or third-party
# SHOULD NOT use it.
#
# This software is provided by the copyright holders and contributors "as is" and
# any express or implied warranties, including, but not limited to, the implied
# warranties of merchantability and fitness for a particular purpose are disclaimed.
# In no event shall copyright holders or contributors be liable for any direct,
# indirect, incidental, special, exemplary, or consequential damages
# (including, but not limited to, procurement of substitute goods or services;
# loss of use, data, or profits; or business interruption) however caused
# and on any theory of liability, whether in contract, strict liability,
# or tort (including negligence or otherwise) arising in any way out of
# the use of this software, even if advised of the possibility of such damage.
#
# @Time    : 18-5-25
# @Author  : Gavin.Xu
# @Email   : Gavin.Xu@bmw.com
# @Department: EG-CN-72

import argparse


def set_train_args():
    """
    configure the training arguments

    :return: args
    """
    parser = argparse.ArgumentParser()

    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='number of epochs')

    parser.add_argument('--save_every_epoch', type=int, default=10,
                        help='save model variables for every how many epoch(s)')

    # size of each batch for training dataset parameter
    parser.add_argument('--batch_size', type=int, default=32,
                        help='training set minibatch size')

    # width of context map parameter
    parser.add_argument('--width', type=int, default=1280,
                        help='width of the context map')

    # height of context map parameter
    parser.add_argument('--height', type=int, default=1280,
                        help='height of the context map')

    # number of input trajectory point parameter
    parser.add_argument('--seq_length', type=int, default=64,
                        help='number of input trajectory point')

    # output size of convolution network parameter
    parser.add_argument('--target_size_initial', type=int, default=512,
                        help='output size of cnn')

    # filter size of convolution network parameter
    parser.add_argument('--filter_size', type=int, default=32,
                        help='filter size of cnn')

    # dropout probability parameter for dropout layer in convolution network
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout probability of dropout layer in cnn')

    # number of discrete steps
    parser.add_argument('--discrete_timestamps', type=list, default=[1, 2, 3, 4, 5, 6],
                        help='List of the specific timestamp to be predict')

    # predict frequency
    parser.add_argument('--frequency', type=int, default=20,
                        help='predict frequency')

    # size of feature, i.e. a trajectory point representation
    parser.add_argument('--feature_size', type=int, default=2,
                        help='size of a trajectory point')

    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN output and hidden state')

    # Number of layers in RNN parameter
    parser.add_argument('--n_layers', type=int, default=2,
                        help='number of layers for each cell in the RNN')

    # Dropout probability parameter
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')

    # learning rate for RNN parameter
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate for RNN')

    # momentum for RNN
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for RNN')

    # mode
    parser.add_argument('--mode', type=str, default='train',
                        help='train or infer')

    # directory for train summary FileWriter
    parser.add_argument('--train_summary', type=str, default='./train/',
                        help='train summary FileWriter')

    # directory for val summary FileWriter
    parser.add_argument('--val_summary', type=str, default='./val/',
                        help='val summary FileWriter')

    args = parser.parse_args()
    return args
