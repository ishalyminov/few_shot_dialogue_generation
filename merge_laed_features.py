import os
from argparse import ArgumentParser
import pickle
from collections import defaultdict

import numpy as np


def load_features(in_folder, process_seed_data=False):
    result = {}
    with open(os.path.join(in_folder, 'dialogs.pkl')) as dialogs_in:
        result['dialogs.pkl'] = pickle.load(dialogs_in)
    if process_seed_data:
        with open(os.path.join(in_folder, 'seed_utts.pkl')) as seed_in:
            result['seed_utts.pkl'] = pickle.load(seed_in)
    return result


def merge_features(in_features):
    assert len(in_features), 'No features to merge'
    
    result = defaultdict(lambda: [])

    assert len(set([len(features_i['dialogs.pkl']) for features_i in features])) == 1, \
           'Got different feature lengths: {}'.format([str(len(features_i['dialogs.pkl'])) for features_i in features])
    for idx in range(len(features[0]['dialogs.pkl'])):
        result['dialogs.pkl'].append(np.concatenate([features_i['dialogs.pkl'][idx] for features_i in features], axis=-1))
    if 'seed_utts.pkl' in in_features[0]:
        result['seed_utts.pkl'] = np.concatenate([features_i['seed_utts.pkl'] for features_i in features], axis=-1)
    return result


def save_features(in_features, out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for key in in_features:
        with open(os.path.join(out_folder, key), 'w') as feature_out:
            pickle.dump(in_features[key], feature_out)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('feature_folders', type=str, nargs='+')
    parser.add_argument('out_folder')
    parser.add_argument('--process_seed_data', default=False, action='store_true')

    args = parser.parse_args()

    features = [load_features(folder_name, args.process_seed_data)
                for folder_name in args.feature_folders]
    merged_features = merge_features(features)
    save_features(merged_features, args.out_folder)

