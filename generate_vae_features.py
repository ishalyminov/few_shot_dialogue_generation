from __future__ import print_function

import argparse
import logging
import os
import sys

import numpy as np
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), 'NeuralDialog-LAED'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'NeuralDialog-ZSDG'))

from laed.dataset import data_loaders
from laed.utils import str2bool, prepare_dirs_loggers, get_time, process_config
from laed.enc2dec.decoders import TEACH_FORCE 
from utils import corpora
from utils.corpora import load_vocab, load_model, load_config

arg_lists = []
parser = argparse.ArgumentParser()
logger = logging.getLogger()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def process_data_feed(model, feed, config):
    features = []
    model.eval()
    feed.epoch_init(config, shuffle=False, verbose=True)
    while True:
        batch = feed.next_batch()
        if batch is None:
            break
        laed_out = model.forward(batch, TEACH_FORCE, config.gen_type, return_latent=True)
        laed_z = laed_out['sample_y']
        features.append(laed_z.data.cpu().numpy())
    return np.array(features).reshape(-1, config.y_size * config.k)


def deflatten_laed_features(in_laed_features, in_dialogs, pad_mode=None):
    pad = np.zeros_like(in_laed_features[0])
    result = []
    start_turn = 0
    for dialog_i in in_dialogs:
        if pad_mode == 'start_end':
            dialog_i_turns = np.concatenate([[pad],
                                             in_laed_features[start_turn: start_turn + len(dialog_i) - 2,:],
                                             [pad]],
                                            axis=0)
            start_turn += len(dialog_i) - 2
        elif pad_mode == 'start':
            dialog_i_turns = np.concatenate([[pad],
                                             in_laed_features[start_turn: start_turn + len(dialog_i) - 1,:]],
                                            axis=0)
            start_turn += len(dialog_i) - 1
        else:
            dialog_i_turns = in_laed_features[start_turn: start_turn + len(dialog_i),:]
            start_turn += len(dialog_i)
        while len(dialog_i) < dialog_i_turns.shape[0]:
            dialog_i_turns = np.delete(dialog_i_turns, (-1), axis=0)
        result.append(dialog_i_turns)
    assert len(in_dialogs) == len(result)
    return result


def main(config):
    laed_config = load_config(config.model)
    laed_config.use_gpu = config.use_gpu
    laed_config = process_config(laed_config)

    setattr(laed_config, 'black_domains', config.black_domains)
    setattr(laed_config, 'black_ratio', config.black_ratio)
    setattr(laed_config, 'include_domain', True)
    setattr(laed_config, 'include_example', False)
    setattr(laed_config, 'include_state', True)
    setattr(laed_config, 'entities_file', 'NeuralDialog-ZSDG/data/stanford/kvret_entities.json')
    setattr(laed_config, 'action_match', True)
    setattr(laed_config, 'batch_size', config.batch_size)
    setattr(laed_config, 'data_dir', config.data_dir)
    setattr(laed_config, 'include_eod', False) # for StED model
    setattr(laed_config, 'domain_description', config.domain_description)

    if config.process_seed_data:
        assert config.corpus_client[:3] == 'Zsl', 'Incompatible coprus_client for --process_seed_data flag'
    corpus_client = getattr(corpora, config.corpus_client)(laed_config)
    if config.vocab:
        corpus_client.vocab, corpus_client.rev_vocab, corpus_client.unk_token = load_vocab(config.vocab)
    prepare_dirs_loggers(config, os.path.basename(__file__))

    dial_corpus = corpus_client.get_corpus()
    # train_dial, valid_dial, test_dial = dial_corpus['train'], dial_corpus['valid'], dial_corpus['test']
    # all_dial = train_dial + valid_dial + test_dial
    # all_utts = reduce(lambda x, y: x + y, all_dial, [])

    model = load_model(config.model, config.model_name, config.model_type, corpus_client=corpus_client)

    if config.use_gpu:
        model.cuda()

    for dataset_name in ['train', 'valid', 'test']:
        dataset = dial_corpus[dataset_name]
        feed_data = dataset if config.model_type == 'dialog' else reduce(lambda x, y: x + y, dataset, [])

        # create data loader that feed the deep models
        if config.process_seed_data:
            seed_utts = corpus_client.get_seed_responses(utt_cnt=len(corpus_client.domain_descriptions))
        main_feed = getattr(data_loaders, config.data_loader)("Test", feed_data, laed_config)

        features = process_data_feed(model, main_feed, laed_config)
        if config.data_loader == 'SMDDialogSkipLoader':
            pad_mode = 'start_end'
        elif config.data_loader == 'SMDDataLoader':
            pad_mode = 'start'
        else:
            pad_mode = None
        features = deflatten_laed_features(features, dataset, pad_mode=pad_mode)
        assert sum(map(len, dataset)) == sum(map(lambda x: x.shape[0], features))

        if not os.path.exists(config.out_folder):
            os.makedirs(config.out_folder)
        with open(os.path.join(config.out_folder, 'dialogs_{}.pkl'.format(dataset_name)), 'w') as result_out:
            pickle.dump(features, result_out)

    if config.process_seed_data:
        seed_utts = corpus_client.get_seed_responses(utt_cnt=len(corpus_client.domain_descriptions))
        seed_feed = data_loaders.PTBDataLoader("Seed", seed_utts, laed_config)
        seed_features = process_data_feed(model, seed_feed, laed_config)
        with open(os.path.join(config.out_folder, 'seed_utts.pkl'), 'w') as result_out:
            pickle.dump(seed_features, result_out)


if __name__ == "__main__":
    # Data
    data_arg = add_argument_group('Data')
    data_arg.add_argument('model')
    data_arg.add_argument('out_folder')
    data_arg.add_argument('--model_name', required=True)
    data_arg.add_argument('--model_type', required=True, help='sent/dialog')
    data_arg.add_argument('--data_dir', nargs='+')
    data_arg.add_argument('--corpus_client', required=True)
    data_arg.add_argument('--data_loader', required=True, help='PTBDataLoader/SMDDataLoader/SMDDialogSkipLoader')
    data_arg.add_argument('--black_domains', nargs='*', default=[])
    data_arg.add_argument('--black_ratio', type=float, default=1.0)
    data_arg.add_argument('--batch_size', default=1)
    data_arg.add_argument('--process_seed_data', default=False, action='store_true')
    data_arg.add_argument('--vocab', default=None)
    data_arg.add_argument('--domain_description', default='annotation')

    # MISC
    misc_arg = add_argument_group('Misc')
    misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
    misc_arg.add_argument('--forward_only', type=str2bool, default=True)
    misc_arg.add_argument('--gen_type', type=str, default='greedy')

    config, unparsed = get_config()
    main(config)

