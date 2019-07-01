import argparse
import json
import os
import pickle

import torch
import numpy as np

from corpora import LAEDBlisCorpus
from laed.models import sent_models, dialog_models


def load_laed_features(in_folder):
    with open(os.path.join(in_folder, 'dialogs.pkl')) as laed_z_dialog_in:
        laed_z_dialog = pickle.load(laed_z_dialog_in)
    if not os.path.exists(os.path.join(in_folder, 'seed_utts.pkl')):
        laed_z_seed = []
    else:
        with open(os.path.join(in_folder, 'seed_utts.pkl')) as laed_z_seed_in:
            laed_z_seed = np.array(pickle.load(laed_z_seed_in))
    return {'dialog': laed_z_dialog, 'seed': laed_z_seed}


def load_config(in_model_folder):
    with open(os.path.join(in_model_folder, 'params.json')) as config_in:
        model_config = json.load(config_in)
        config = argparse.Namespace()
        for key, value in model_config.items():
            setattr(config, key, value)
    return config


def load_model(in_model_folder, in_model_name, in_model_type, config=None, corpus_client=None):
    if not config:
        with open(os.path.join(in_model_folder, 'params.json')) as config_in:
            model_config = json.load(config_in)
            config = argparse.Namespace()
            for key, value in model_config.items():
                setattr(config, key, value)
    if not corpus_client:
        corpus_client = LAEDBlisCorpus(config)
    module = sent_models if in_model_type == 'sent' else dialog_models
 
    laed_model = getattr(module, in_model_name)(corpus_client, config)
    laed_model.load_state_dict(torch.load(os.path.join(in_model_folder, 'model')))

    return laed_model
