from __future__ import print_function

import argparse
import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'NeuralDialog-LAED'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'NeuralDialog-ZSDG'))

from laed.utils import str2bool, prepare_dirs_loggers, get_time, process_config

from utils import corpora
from utils.corpora import save_vocab

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('data_dir', nargs='+')
data_arg.add_argument('corpus_client', help='LAEDBlisCorpus/ZslStanfordCorpus/SimDialCorpus')
data_arg.add_argument('output_file')
data_arg.add_argument('--black_domains', nargs='*', default=[])
data_arg.add_argument('--black_ratio', type=float, default=0.0)
data_arg.add_argument('--include_domain', default=True)
data_arg.add_argument('--exclude_domains', nargs='*', default=[])
data_arg.add_argument('--log_dir', type=str, default='logs')

net_arg = add_argument_group('Network')
net_arg.add_argument('--y_size', type=int, default=20)  # number of discrete variables
net_arg.add_argument('--k', type=int, default=10)  # number of classes for each variable
net_arg.add_argument('--rnn_cell', type=str, default='gru')
net_arg.add_argument('--embed_size', type=int, default=200)
net_arg.add_argument('--utt_type', type=str, default='rnn')
net_arg.add_argument('--enc_cell_size', type=int, default=512)
net_arg.add_argument('--dec_cell_size', type=int, default=512)
net_arg.add_argument('--bi_enc_cell', type=str2bool, default=False)
net_arg.add_argument('--max_utt_len', type=int, default=40)
net_arg.add_argument('--max_dec_len', type=int, default=40)
net_arg.add_argument('--max_vocab_cnt', type=int, default=10000)
net_arg.add_argument('--num_layer', type=int, default=1)
net_arg.add_argument('--use_attn', type=str2bool, default=False)
net_arg.add_argument('--attn_type', type=str, default='cat')
net_arg.add_argument('--use_mutual', type=str2bool, default=True)
net_arg.add_argument('--use_reg_kl', type=str2bool, default=True)


# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--op', type=str, default='adam')
train_arg.add_argument('--backward_size', type=int, default=5)
train_arg.add_argument('--step_size', type=int, default=1)
train_arg.add_argument('--grad_clip', type=float, default=3.0)
train_arg.add_argument('--init_w', type=float, default=0.08)
train_arg.add_argument('--init_lr', type=float, default=0.001)
train_arg.add_argument('--momentum', type=float, default=0.0)
train_arg.add_argument('--lr_hold', type=int, default=1)
train_arg.add_argument('--lr_decay', type=float, default=0.6)
train_arg.add_argument('--dropout', type=float, default=0.3)
train_arg.add_argument('--improve_threshold', type=float, default=0.996)
train_arg.add_argument('--patient_increase', type=float, default=2.0)
train_arg.add_argument('--early_stop', type=str2bool, default=True)
train_arg.add_argument('--max_epoch', type=int, default=50)
train_arg.add_argument('--entities_file',
                       type=str,
                       default='NeuralDialog-ZSDG/data/stanford/kvret_entities.json')


# MISC
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--save_model', type=str2bool, default=True)
misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
misc_arg.add_argument('--print_step', type=int, default=500)
misc_arg.add_argument('--fix_batch', type=str2bool, default=False)
misc_arg.add_argument('--train_prior', type=str2bool, default=False)
misc_arg.add_argument('--ckpt_step', type=int, default=2000)
misc_arg.add_argument('--batch_size', type=int, default=30)
misc_arg.add_argument('--preview_batch_num', type=int, default=1)
misc_arg.add_argument('--gen_type', type=str, default='greedy')
misc_arg.add_argument('--avg_type', type=str, default='word')
misc_arg.add_argument('--beam_size', type=int, default=10)
misc_arg.add_argument('--forward_only', type=str2bool, default=False)
data_arg.add_argument('--load_sess', type=str, default="2018-02-04T01-20-45")
logger = logging.getLogger()


def main(config):
    corpus_client = getattr(corpora, config.corpus_client)(config)
    save_vocab(corpus_client.vocab, config.output_file)


if __name__ == "__main__":
    config, unparsed = get_config()
    config = process_config(config)
    main(config)

