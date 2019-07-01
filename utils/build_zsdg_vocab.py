import json
from argparse import ArgumentParser
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'NeuralDialog-LAED'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'NeuralDialog-ZSDG'))

from zsdg.utils import str2bool
from corpora import ZslBlisCorpus

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('out_file')

    parser.add_argument('--data_dir',
                          nargs='+',
                          default=['data/crowdsourced_task-oriented_dialogues/blis_collected_dialogues.json'])
    parser.add_argument('--max_vocab_cnt', type=int, default=10000)
    parser.add_argument('--max_utt_len', type=int, default=20)
    parser.add_argument('--black_domains', type=list, default=['RESTAURANT_PICKER'])
    parser.add_argument('--black_ratio', type=float, default=1.0)
    parser.add_argument('--target_example_cnt', type=int, default=150)
    parser.add_argument('--entities_file',
                        type=str,
                        default='NeuralDialog-ZSDG/data/stanford/kvret_entities.json')
    parser.add_argument('--include_domain', type=str2bool, default=True)


    config, _ = parser.parse_known_args()

    corpus_client = ZslBlisCorpus(config)

    with open(config.out_file, 'w') as vocab_out:
        json.dump(corpus_client.vocab, vocab_out)
