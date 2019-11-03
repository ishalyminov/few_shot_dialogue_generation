from __future__ import print_function

import argparse
import os
import torch
import sys
import random

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'NeuralDialog-ZSDG'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'NeuralDialog-LAED'))

from utils import data_loaders
from zsdg.main import train, validate
from zsdg import hred_utils
from zsdg.utils import str2bool, prepare_dirs_loggers, get_time, process_config
from zsdg import evaluators

from models.models import LAPtrHRED, ZeroShotLAPtrHRED
from zsdg.models.models import PtrHRED, ZeroShotPtrHRED
from utils import corpora

arg_lists = []
parser = argparse.ArgumentParser()


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def get_corpus_client(config):
    if len(config.laed_z_folders):
        return corpora.LAZslStanfordCorpus
    else:
        return corpora.ZslStanfordCorpus

def get_data_loader(config):
    if len(config.laed_z_folders):
        return data_loaders.ZslLASMDDialDataLoader
    else:
        return data_loaders.ZslSMDDialDataLoader


def get_model(config, train_client):
    if len(config.laed_z_folders):
        if config.action_match:
            if config.use_ptr:
                model = ZeroShotLAPtrHRED(train_client, config)
            else:
                raise NotImplementedError()
        else:
            model = LAPtrHRED(train_client, config)
    else:
        if config.action_match:
            if config.use_ptr:
                model = ZeroShotPtrHRED(train_client, config)
            else:
                raise NotImplementedError()
        else:
            model = PtrHRED(train_client, config) 
    return model

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--data_dir',
                      nargs='+',
                      default=['NeuralDialog-ZSDG/data/stanford'])
data_arg.add_argument('--log_dir', type=str, default='logs')
data_arg.add_argument('--laed_z_folders', nargs='+', default=[])

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--rnn_cell', type=str, default='lstm')
net_arg.add_argument('--embed_size', type=int, default=200)
net_arg.add_argument('--utt_type', type=str, default='rnn')
net_arg.add_argument('--utt_cell_size', type=int, default=256)
net_arg.add_argument('--ctx_cell_size', type=int, default=512)
net_arg.add_argument('--dec_cell_size', type=int, default=512)
net_arg.add_argument('--bi_ctx_cell', type=str2bool, default=False)
net_arg.add_argument('--max_utt_len', type=int, default=20)
net_arg.add_argument('--max_dec_len', type=int, default=40)
net_arg.add_argument('--num_layer', type=int, default=1)
net_arg.add_argument('--use_attn', type=str2bool, default=True)
net_arg.add_argument('--attn_type', type=str, default='cat')
net_arg.add_argument('--max_vocab_cnt', type=int, default=10000)
net_arg.add_argument('--vocab', type=str, default=None)

train_arg = add_argument_group('Training')
train_arg.add_argument('--op', type=str, default='adam')
train_arg.add_argument('--backward_size', type=int, default=14)
train_arg.add_argument('--step_size', type=int, default=2)
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
train_arg.add_argument('--preview_batch_num', type=int, default=50)
train_arg.add_argument('--include_domain', type=str2bool, default=True)
train_arg.add_argument('--include_example', type=str2bool, default=False)
train_arg.add_argument('--include_state', type=str2bool, default=True)
train_arg.add_argument('--random_seed', type=int, default=271)
train_arg.add_argument('--dd_loss_coef', type=float, default=0.0)

# MISC
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--save_model', type=str2bool, default=True)
misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
misc_arg.add_argument('--print_step', type=int, default=100)
misc_arg.add_argument('--ckpt_step', type=int, default=400)
misc_arg.add_argument('--batch_size', type=int, default=4)
misc_arg.add_argument('--gen_type', type=str, default='greedy')
misc_arg.add_argument('--avg_type', type=str, default='word')
misc_arg.add_argument('--beam_size', type=int, default=20)

# KEY PARAMETERS

# decide which domains are excluded from the training
train_arg.add_argument('--black_domains', type=str, nargs='*', default=['navigate'])
train_arg.add_argument('--black_ratio', type=float, default=1.0)
train_arg.add_argument('--target_example_cnt', type=int, default=150)
train_arg.add_argument('--source_example_cnt', type=int, default=150)
train_arg.add_argument('--domain_description', type=str, default='nlu')
train_arg.add_argument('--entities_file',
                       type=str,
                       default='NeuralDialog-ZSDG/data/stanford/kvret_entities.json')

# Which model is used
net_arg.add_argument('--action_match', type=str2bool, default=True)
net_arg.add_argument('--use_ptr', type=str2bool, default=True)

# Where to load existing model
misc_arg.add_argument('--forward_only', default=False, action='store_true')
misc_arg.add_argument('--load_sess', type=str, default="ENTER_YOUR_PATH_HERE")


def main(config):
    prepare_dirs_loggers(config, os.path.basename(__file__))

    corpus_client_class = get_corpus_client(config)
    train_client = corpus_client_class(config)
    train_corpus = train_client.get_corpus()
    train_dial, valid_dial, test_dial = train_corpus['train'], train_corpus['valid'], train_corpus['test']

    evaluator = evaluators.BleuEntEvaluator("SMD", train_client.ent_metas)

    # create data loader that feed the deep models
    data_loader_class = get_data_loader(config)
    train_feed = data_loader_class("Train", train_dial, config)
    valid_feed = data_loader_class("Valid", valid_dial, config)
    test_feed = data_loader_class("Test", test_dial, config)

    model = get_model(config, train_client)

    if config.forward_only:
        session_dir = os.path.join(config.log_dir, config.load_sess)
        test_file = os.path.join(session_dir, "{}-test-{}.txt".format(get_time(), config.gen_type))
        model_file = os.path.join(config.log_dir, config.load_sess, "model")
    else:
        session_dir = config.session_dir
        test_file = os.path.join(session_dir,
                                 "{}-test-{}.txt".format(get_time(), config.gen_type))
        model_file = os.path.join(config.session_dir, "model")

    if config.use_gpu:
        model.cuda()
    if not config.forward_only:
        try:
            train(model, train_feed, valid_feed, test_feed, config, evaluator, gen=hred_utils.generate)
        except KeyboardInterrupt:
            print("Training stopped by keyboard.")

    config.batch_size = 20
    model.load_state_dict(torch.load(model_file))

    # run the model on the test dataset.
    validate(model, test_feed, config)

    with open(os.path.join(test_file), "wb") as f:
        hred_utils.generate(model, test_feed, config, evaluator, num_batch=None, dest_f=f)


if __name__ == "__main__":
    config, unparsed = get_config()
    # config = process_config(config)
    fix_random_seed(config.random_seed)
    main(config)

