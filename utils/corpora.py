import json
import logging
import os
from collections import Counter, defaultdict
import argparse
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
import torch

from laed.utils import Pack, get_tokenize
from laed.dataset.corpora import *
from zsdg.dataset.corpora import *
from laed.models import sent_models, dialog_models


def save_vocab(in_vocab, in_vocab_file):
    with open(in_vocab_file, 'w') as vocab_out:
        json.dump(in_vocab, vocab_out)


def load_vocab(in_vocab_file):
    with open(in_vocab_file) as vocab_in:
        vocab = json.load(vocab_in)
    rev_vocab = {t: idx for idx, t in enumerate(vocab)}
    unk_id = rev_vocab[UNK]
    return vocab, rev_vocab, unk_id


def _flatten_nlu_dict(self, in_dict):
    result = []
    for key in sorted(in_dict.keys()):
        if type(in_dict[key]) == list:
            for value in in_dict[key]:
                result += [key, value]
        else:
            result += [key, in_dict[key]]
    return result


def load_laed_features(in_folder):
    dialog_mapping = {}
    for dataset in ['train', 'valid', 'test']:
        with open(os.path.join(in_folder, 'dialogs_{}.pkl'.format(dataset)), 'rb') as laed_z_dialog_in:
            laed_z_dialog = pickle.load(laed_z_dialog_in)
            dialog_mapping[dataset] = laed_z_dialog
    if not os.path.exists(os.path.join(in_folder, 'seed_utts.pkl')):
        laed_z_seed = []
    else:
        with open(os.path.join(in_folder, 'seed_utts.pkl', 'rb')) as laed_z_seed_in:
            laed_z_seed = np.array(pickle.load(laed_z_seed_in))
    return {'dialog': dialog_mapping, 'seed': laed_z_seed}


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


class LAEDBlisCorpus(object):
    logger = logging.getLogger()

    def __init__(self, config):
        self.config = config
        self._path = config.data_dir[0]
        self.max_utt_len = config.max_utt_len
        self.tokenize = get_tokenize()
        self.corpus = self._read_file(self._path)
        self.train_corpus, devtest = train_test_split(self.corpus,
                                                      test_size=0.2,
                                                      random_state=271)
        self.valid_corpus, self.test_corpus = train_test_split(devtest,
                                                               test_size=0.5,
                                                               random_state=271)
        if hasattr(self.config, 'vocab') and self.config.vocab:
            self.vocab, self.rev_vocab, self.unk_id = load_vocab(self.config.vocab)
        else:
            self._build_vocab(self.config.max_vocab_cnt)
        print("Done loading corpus")

    def _read_file(self, path):
        data = []
        for filename in os.listdir(os.path.join(path, 'dialogues')):
            with open(os.path.join(path, 'dialogues', filename)) as domain_in:
                for line in domain_in:
                    data.append(json.loads(line))
        return self._process_dialogs(data, exclude_domains=self.config.exclude_domains)

    def _process_dialogs(self, data, exclude_domains=[]):
        new_dialog = []
        bod_utt = [BOS, BOD, EOS]
        eod_utt = [BOS, EOD, EOS]
        all_lens = []
        all_dialog_lens = []
        speaker_map = {'bot': SYS, 'user': USR}
        for raw_dialog in data:
            domain = raw_dialog['domain']
            if domain in exclude_domains:
                continue
            dialog = [Pack(utt=bod_utt, speaker=0, meta=None)]

            for utt_idx, utt in enumerate(raw_dialog['turns']):
                author_type = 'bot' if utt_idx % 2 == 0 else 'user'
                if self.config.include_domain:
                    utt = [BOS, speaker_map[author_type], domain] + self.tokenize(utt) + [EOS]
                else:
                    utt = [BOS, speaker_map[author_type]] + self.tokenize(utt) + [EOS]
                all_lens.append(len(utt))
                dialog.append(Pack(utt=utt, speaker=speaker_map[author_type]))

            if not hasattr(self.config, 'include_eod') or self.config.include_eod:
                dialog.append(Pack(utt=eod_utt, speaker=0))

            all_dialog_lens.append(len(dialog))
            new_dialog.append(dialog)

        print("Max utt len %d, mean utt len %.2f" % (np.max(all_lens), float(np.mean(all_lens))))
        print("Max dialog len %d, mean dialog len %.2f" % (np.max(all_dialog_lens),
                                                           float(np.mean(all_dialog_lens))))
        return new_dialog

    def _build_vocab(self, max_vocab_cnt):
        all_words = []
        for dialog in self.train_corpus:
            for turn in dialog:
                all_words.extend(turn.utt)

        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
        vocab_count = vocab_count[0: max_vocab_cnt]

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at "
              "cut_off %d OOV rate %f" % (len(self.train_corpus),
                                          len(self.valid_corpus),
                                          len(self.test_corpus),
                                          raw_vocab_size,
                                          len(vocab_count),
                                          vocab_count[-1][1],
                                          float(discard_wc) / len(all_words)))

        self.vocab = [PAD, UNK, SYS, USR] + [t for t, cnt in vocab_count]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab[UNK]

    def _sent2id(self, sent):
        return [self.rev_vocab.get(t, self.unk_id) for t in sent]

    def _to_id_corpus(self, data):
        results = []
        for dialog in data:
            temp = []
            # convert utterance and feature into numeric numbers
            for turn in dialog:
                id_turn = Pack(utt=self._sent2id(turn.utt),
                               speaker=turn.speaker,
                               meta=turn.get('meta'))
                temp.append(id_turn)
            results.append(temp)
        return results

    def get_corpus(self):
        id_train = self._to_id_corpus(self.train_corpus)
        id_valid = self._to_id_corpus(self.valid_corpus)
        id_test = self._to_id_corpus(self.test_corpus)
        return Pack(train=id_train, valid=id_valid, test=id_test)


class ZslBlisCorpus(object):
    logger = logging.getLogger(__name__)

    def __init__(self, config):
        self.config = config
        self._path = config.data_dir[0]
        if isinstance(self._path, list):
            self._path = self._path[0]
        self.max_utt_len = config.max_utt_len
        self.tokenize = get_tokenize()
        self.black_domains = config.black_domains
        self.black_ratio = config.black_ratio
        self.corpus = self._read_file(self._path)
        self.train_corpus, devtest = train_test_split(self.corpus, test_size=0.2, random_state=271)
        self.valid_corpus, self.test_corpus = train_test_split(devtest,
                                                               test_size=0.5,
                                                               random_state=271)
        #  TODO: update slot/value map relevant for Maluuba
        with open(self.config.entities_file, 'rb') as f:
            self.ent_metas = json.load(f)
        self.domain_descriptions = self._read_domain_descriptions(os.path.dirname(self._path))
        if hasattr(self.config, 'vocab') and self.config.vocab:
            self.vocab, self.rev_vocab, self.unk_id = load_vocab(self.config.vocab)
        else:
            self._build_vocab(self.config.max_vocab_cnt)
        print("Done loading corpus")

    def _read_domain_descriptions(self, path):
        # read all domains
        seed_responses = []
        speaker_map = {'assistant': SYS, 'driver': USR}

        def _read_file(domain):
            with open(os.path.join(path, 'domain_descriptions/{}.tsv'.format(domain)), 'rb') as f:
                lines = f.readlines()
                for l in lines[1:]:
                    tokens = l.split('\t')
                    if tokens[2] == "":
                        break
                    utt = tokens[1]
                    speaker = tokens[0]
                    action = tokens[3]
                    if self.config.include_domain:
                        utt = [BOS, speaker_map[speaker], domain] + self.tokenize(utt) + [EOS]
                        action = [BOS, speaker_map[speaker], domain] + self.tokenize(action) + [EOS]
                    else:
                        utt = [BOS, speaker_map[speaker]] + self.tokenize(utt) + [EOS]
                        action = [BOS, speaker_map[speaker]] + self.tokenize(action) + [EOS]

                    seed_responses.append(Pack(domain=domain, speaker=speaker,
                                               utt=utt, actions=action))

        _read_file('navigate')
        _read_file('schedule')
        _read_file('weather')
        return seed_responses

    def _read_file(self, path):
        data = []
        for filename in os.listdir(os.path.join(path, 'dialogues')):
            with open(os.path.join(path, 'dialogues', filename)) as domain_in:
                for line in domain_in:
                    data.append(json.loads(line))
        return self._process_dialogs(data)

    def _process_dialogs(self, data):
        new_dialog = []
        bod_utt = [BOS, BOD, EOS]
        all_lens = []
        all_dialog_lens = []
        speaker_map = {'bot': SYS, 'user': USR}
        for raw_dialog in data:
            domain = raw_dialog['domain']
            dialog = [Pack(utt=bod_utt, speaker=USR, meta=None, domain=domain)]

            for utt_idx, utt in enumerate(raw_dialog['turns']):
                speaker = speaker_map['bot' if utt_idx % 2 == 0 else 'user']
                if self.config.include_domain:
                    utt = [BOS, speaker, domain] + self.tokenize(utt) + [EOS]
                else:
                    utt = [BOS, speaker] + self.tokenize(utt) + [EOS]

                all_lens.append(len(utt))
                if speaker == SYS:
                    dialog.append(
                        Pack(utt=utt, speaker=speaker, slots={}, domain=domain, kb={}))
                else:
                    dialog.append(Pack(utt=utt, speaker=speaker, slots={}, domain=domain))

            all_dialog_lens.append(len(dialog))
            new_dialog.append(dialog)

        print("Max utt len %d, mean utt len %.2f" % (
            np.max(all_lens), float(np.mean(all_lens))))
        print("Max dialog len %d, mean dialog len %.2f" % (
            np.max(all_dialog_lens), float(np.mean(all_dialog_lens))))
        return new_dialog

    def _build_vocab(self, max_vocab_cnt):
        all_words = []
        for dialog in self.train_corpus:
            for turn in dialog:
                all_words.extend(turn.utt)

        for resp in self.domain_descriptions:
            all_words.extend(resp.actions)

        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.valid_corpus),
                 len(self.test_corpus),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1],
                 float(discard_wc) / len(all_words)))

        vocab_count = vocab_count[:max_vocab_cnt]
        self.vocab = [PAD, UNK, SYS, USR] + [t for t, cnt in vocab_count]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab[UNK]

    def _sent2id(self, sent):
        return [self.rev_vocab.get(t, self.unk_id) for t in sent]

    def _to_id_corpus(self, name, data, use_black_list):
        results = []
        kick_cnt = 0
        domain_cnt = []
        for dialog in data:
            if len(dialog) < 1:
                continue
            domain = dialog[0].domain
            should_filter = np.random.rand() < self.black_ratio
            if use_black_list and self.black_domains \
                    and domain in self.black_domains \
                    and should_filter:
                kick_cnt += 1
                continue
            temp = []
            # convert utterance and feature into numeric numbers
            for turn in dialog:
                id_turn = Pack(utt=self._sent2id(turn.utt),
                               speaker=turn.speaker,
                               domain=turn.domain,
                               domain_id=self.rev_vocab[domain])
                               #kb=[self._sent2id(item) for item in turn.get('kb', [])])
                temp.append(id_turn)

            results.append(temp)
            domain_cnt.append(domain)
        self.logger.info("Filter {} samples from {}".format(kick_cnt, name))
        self.logger.info(Counter(domain_cnt).most_common())
        return results

    def get_corpus(self):
        id_train = self._to_id_corpus("Train", self.train_corpus, use_black_list=True)
        id_valid = self._to_id_corpus("Valid", self.valid_corpus, use_black_list=False)
        id_test = self._to_id_corpus("Test", self.test_corpus, use_black_list=False)
        return Pack(train=id_train, valid=id_valid, test=id_test)

    def get_seed_responses(self, utt_cnt=100):
        domain_seeds = defaultdict(list)
        all_domains = []
        if utt_cnt == 0 or self.config.action_match is False:
            return []

        for resp in self.domain_descriptions:
            resp_copy = resp.copy()
            resp_copy['utt'] = self._sent2id(resp.utt)
            resp_copy['actions'] = self._sent2id(resp.actions)
            resp_copy['domain_id'] = self.rev_vocab[resp.domain]
            if len(domain_seeds[resp.domain]) >= utt_cnt:
                continue

            domain_seeds[resp.domain].append(resp_copy)
            all_domains.append(resp.domain)

        seed_responses = []
        for v in domain_seeds.values():
            seed_responses.extend(v)

        self.logger.info("Collected {} extra samples".format(len(seed_responses)))
        self.logger.info(Counter(all_domains).most_common())
        return seed_responses


class ZslStanfordCorpus(object):
    logger = logging.getLogger(__name__)

    def __init__(self, config):
        self.config = config
        self._path = config.data_dir[0]
        self.max_utt_len = config.max_utt_len
        self.tokenize = get_tokenize()
        self.black_domains = config.black_domains
        self.black_ratio = config.black_ratio
        self.speaker_map = {'assistant': SYS, 'driver': USR}
        self.domain_descriptions = []

        self.train_corpus = self._read_file(os.path.join(self._path, 'kvret_train_public.json'))
        self.valid_corpus = self._read_file(os.path.join(self._path, 'kvret_dev_public.json'))
        self.test_corpus = self._read_file(os.path.join(self._path, 'kvret_test_public.json'))

        self.domains = set([dialog[0].domain for dialog in self.train_corpus])

        with open(os.path.join(self._path, 'kvret_entities.json'), 'rb') as f:
            self.ent_metas = json.load(f)
            if self.config.lowercase:
                self.ent_metas = self._lowercase_json(self.ent_metas)

        if self.config.domain_description == 'annotated':
            self.domain_descriptions = self._read_domain_descriptions_annotated(self._path)
        if self.config.domain_description == 'kb':
            self.domain_descriptions = self._read_domain_descriptions_annotated(self._path)

        self._build_vocab()
        print("Done loading corpus")

    def _lowercase_json(self, in_json):
        return json.loads(json.dumps(in_json).lower())

    def _read_file(self, path):
        with open(path, 'rb') as f:
            data = json.load(f)
            if self.config.lowercase:
                data = self._lowercase_json(data)

        return self._process_dialog(data)

    def _read_domain_descriptions_annotated(self, path):
        # read all domains
        seed_responses = []

        def _read_file(domain):
            with open(os.path.join(path, 'domain_descriptions/{}.tsv'.format(domain)), 'rb') as f:
                lines = f.readlines()
                for l in lines[1:]:
                    tokens = l.lower().split('\t')
                    if tokens[2] == "":
                        break
                    utt = tokens[1]
                    speaker = tokens[0]
                    action = tokens[3]
                    if self.config.include_domain:
                        utt = [BOS, self.speaker_map[speaker], domain] + self.tokenize(utt) + [EOS]
                        action = [BOS, self.speaker_map[speaker], domain] + self.tokenize(action) + [EOS]
                    else:
                        utt = [BOS, self.speaker_map[speaker]] + self.tokenize(utt) + [EOS]
                        action = [BOS, self.speaker_map[speaker]] + self.tokenize(action) + [EOS]

                    seed_responses.append(Pack(domain=domain,
                                               speaker=speaker,
                                               utt=utt,
                                               actions=action))

        _read_file('navigate')
        _read_file('schedule')
        _read_file('weather')
        return seed_responses

    def _read_domain_descriptions_kb(self, path):
        # read all domains
        seed_responses = []

        with open(os.path.join(path, 'kb_seed_data.json')) as kb_data_in:
            kb_data = json.load(kb_data_in)
        for domain, turns in kb_data.items():
            for turn in turns:
                if self.config.include_domain:
                    utt = [BOS, self.speaker_map[turn['agent']], domain] + self.tokenize(turn['utterance']) + [EOS]
                    kb = [BOS, self.speaker_map[turn['agent']], domain] + self.tokenize(turn['kb']) + [EOS]
                else:
                    utt = [BOS, self.speaker_map[turn['agent']]] + self.tokenize(turn['utterance']) + [EOS]
                    kb = [BOS, self.speaker_map[turn['agent']]] + self.tokenize(turn['kb']) + [EOS]

                seed_responses.append(Pack(domain=domain, speaker=turn['agent'], utt=utt, actions=kb))
        return seed_responses

    def _process_dialog(self, data):
        new_dialog = []
        all_lens = []
        all_dialog_lens = []
        for raw_dialog in data:
            domain = raw_dialog['scenario']['task']['intent']
            kb_items = []
            if raw_dialog['scenario']['kb']['items'] is not None:
                for item in raw_dialog['scenario']['kb']['items']:
                    kb_items.append([KB]+self.tokenize(" ".join(["{} {}".format(k, v) for k, v in item.items()])))

            dialog = [Pack(utt=[BOS, domain, BOD, EOS], speaker=USR, slots=None, domain=domain)]
            for turn in raw_dialog['dialogue']:
                utt = turn['data']['utterance']
                slots = turn['data'].get('slots')
                speaker = self.speaker_map[turn['turn']]
                if self.config.include_domain:
                    utt = [BOS, speaker, domain] + self.tokenize(utt) + [EOS]
                else:
                    utt = [BOS, speaker] + self.tokenize(utt) + [EOS]

                all_lens.append(len(utt))
                if speaker == SYS:
                    dialog.append(Pack(utt=utt, speaker=speaker, slots=slots, domain=domain, kb=kb_items))
                else:
                    dialog.append(Pack(utt=utt, speaker=speaker, slots=slots, domain=domain, kb=[]))
                self._process_domain_description(turn, domain)

            all_dialog_lens.append(len(dialog))
            new_dialog.append(dialog)

        print("Max utt len %d, mean utt len %.2f" % (
            np.max(all_lens), float(np.mean(all_lens))))
        print("Max dialog len %d, mean dialog len %.2f" % (
            np.max(all_dialog_lens), float(np.mean(all_dialog_lens))))
        return new_dialog

    def _process_domain_description(self, in_turn, in_domain):
        utt = in_turn['data']['utterance']
        speaker = self.speaker_map[in_turn['turn']]

        action = []
        nlu_dict = in_turn['data'].get('nlu', {}).get('annotations', {})
        if 'ner' in nlu_dict:
            action += flatten_nlu_dict(nlu_dict['ner'])
        if 'entity_linking' in nlu_dict:
            for key, value in nlu_dict['entity_linking'].items():
                actual_value = value
                if type(value) == list:
                    actual_value = value[0]
                action.append(actual_value['entity'])
        if 'SUTime' in nlu_dict:
            for item in nlu_dict['SUTime']:
                action.append(item['text'])
        action = ' '.join(action)
        empty_action = not len(action.strip())
        if self.config.include_domain:
            utt = [BOS, speaker, in_domain] + self.tokenize(utt) + [EOS]
            action = [BOS, speaker, in_domain] + self.tokenize(action) + [EOS]
        else:
            utt = [BOS, speaker] + self.tokenize(utt) + [EOS]
            action = [BOS, speaker] + self.tokenize(action) + [EOS]
        self.domain_descriptions.append(Pack(domain=in_domain,
                                             speaker=speaker,
                                             utt=utt,
                                             actions=action,
                                             empty_action=empty_action))

    def _build_vocab(self):
        all_words = []
        for dialog in self.train_corpus:
            for turn in dialog:
                all_words.extend(turn.utt)
                for item in turn.get('kb', []):
                    all_words.extend(item)

        for resp in self.domain_descriptions:
            all_words.extend(resp.actions)

        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count])

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.valid_corpus),
                 len(self.test_corpus),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1],
                 float(discard_wc) / len(all_words)))

        self.vocab = [PAD, UNK, SYS, USR] + [t for t, cnt in vocab_count]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab[UNK]

    def _sent2id(self, sent):
        return [self.rev_vocab.get(t, self.unk_id) for t in sent]

    def _to_id_corpus(self, name, data, use_black_list, domains):
        results = []
        kick_cnt = 0
        domain_cnt = []
        for dialog in data:
            if len(dialog) < 1:
                continue
            domain = dialog[0].domain
            if domain not in domains:
                continue
            should_filter = np.random.rand() < self.black_ratio
            if use_black_list and self.black_domains \
                    and domain in self.black_domains \
                    and should_filter:
                kick_cnt += 1
                continue
            temp = []
            # convert utterance and feature into numeric numbers
            for turn in dialog:
                id_turn = Pack(utt=self._sent2id(turn.utt),
                               utt_raw=turn.utt,
                               speaker=turn.speaker,
                               domain=turn.domain,
                               domain_id=self.rev_vocab[domain],
                               meta=turn.get('meta'),
                               kb=[self._sent2id(item) for item in turn.get('kb', [])],
                               kb_raw=turn.get('kb', []))
                temp.append(id_turn)

            results.append(temp)
            domain_cnt.append(domain)
        self.logger.info("Filter {} samples from {}".format(kick_cnt, name))
        self.logger.info(Counter(domain_cnt).most_common())
        return results

    def get_corpus(self, domains=None):
        real_domains = domains if domains else self.domains
        id_train = self._to_id_corpus("Train",
                                      self.train_corpus,
                                      use_black_list=True,
                                      domains=real_domains)
        id_valid = self._to_id_corpus("Valid",
                                      self.valid_corpus,
                                      use_black_list=False,
                                      domains=real_domains)
        id_test = self._to_id_corpus("Test",
                                     self.test_corpus,
                                     use_black_list=False,
                                     domains=real_domains)
        return Pack(train=id_train, valid=id_valid, test=id_test)

    def get_seed_responses(self, utt_cnt=defaultdict(lambda: 100)):
        domain_seeds = defaultdict(list)
        all_domains = []
        if sum(utt_cnt.values()) == 0 or self.config.action_match is False:
            return []

        for resp in self.domain_descriptions:
            if resp.empty_action:
                continue
            resp_copy = resp.copy()
            resp_copy['utt_raw'] = resp.utt
            resp_copy['actions_raw'] = resp.actions
            resp_copy['domain_raw'] = resp.domain
            resp_copy['utt'] = self._sent2id(resp.utt)
            resp_copy['actions'] = self._sent2id(resp.actions)
            resp_copy['domain_id'] = self.rev_vocab[resp.domain]
            if len(domain_seeds[resp.domain]) >= utt_cnt[resp.domain]:
                continue

            domain_seeds[resp.domain].append(resp_copy)
            all_domains.append(resp.domain)

        seed_responses = []
        for v in domain_seeds.values():
            seed_responses.extend(v)

        self.logger.info("Collected {} extra samples".format(len(seed_responses)))
        self.logger.info(Counter(all_domains).most_common())
        return seed_responses


class LAZslStanfordCorpus(object):
    logger = logging.getLogger(__name__)

    def __init__(self, config):
        self.config = config
        self._path = config.data_dir[0]
        self.max_utt_len = config.max_utt_len
        self.tokenize = get_tokenize()
        self.black_domains = config.black_domains
        self.black_ratio = config.black_ratio
        self.speaker_map = {'assistant': SYS, 'driver': USR}
        self.domain_descriptions = []

        self._load_laed_z()

        self.train_corpus = self._read_file_with_laed_z(os.path.join(self._path, 'kvret_train_public.json'),
                                                        self.laed_z['train'])
        self.valid_corpus = self._read_file_with_laed_z(os.path.join(self._path, 'kvret_dev_public.json'),
                                                        self.laed_z['valid'])
        self.test_corpus = self._read_file_with_laed_z(os.path.join(self._path, 'kvret_test_public.json'),
                                                       self.laed_z['test'])

        self.domains = set([dialog[0].domain for dialog in self.train_corpus])

        with open(os.path.join(self._path, 'kvret_entities.json'), 'rb') as f:
            self.ent_metas = json.load(f)
            if self.config.lowercase:
                self.ent_metas = self._lowercase_json(self.ent_metas)

        if self.config.domain_description == 'annotated':
            self.domain_descriptions = self._read_domain_descriptions_annotated(self._path)
        if self.config.domain_description == 'kb':
            self.domain_descriptions = self._read_domain_descriptions_kb(self._path)

        self._build_vocab()
        print("Done loading corpus")

    def _lowercase_json(self, in_json):
        return json.loads(json.dumps(in_json).lower())

    def _read_domain_descriptions_annotated(self, path):
        # read all domains
        seed_responses = []

        def _read_file(domain):
            with open(os.path.join(path, 'domain_descriptions/{}.tsv'.format(domain)), 'rb') as f:
                lines = f.readlines()
                for l in lines[1:]:
                    tokens = l.split('\t')
                    if tokens[2] == "":
                        break
                    utt = tokens[1]
                    speaker = tokens[0]
                    action = tokens[3]
                    if self.config.include_domain:
                        utt = [BOS, self.speaker_map[speaker], domain] + self.tokenize(utt) + [EOS]
                        action = [BOS, self.speaker_map[speaker], domain] + self.tokenize(action) + [EOS]
                    else:
                        utt = [BOS, self.speaker_map[speaker]] + self.tokenize(utt) + [EOS]
                        action = [BOS, self.speaker_map[speaker]] + self.tokenize(action) + [EOS]

                    seed_responses.append(Pack(domain=domain,
                                               speaker=speaker,
                                               utt=utt,
                                               actions=action))

        _read_file('navigate')
        _read_file('schedule')
        _read_file('weather')
        return seed_responses

    def _read_file_with_laed_z(self, path, in_laed_z):
        with open(path, 'rb') as f:
            data = json.load(f)
            if self.config.lowercase:
                data = self._lowercase_json(data)

        return self._process_dialog(data, in_laed_z)

    def _process_dialog(self, data, laed_z_data):
        new_dialog = []
        all_lens = []
        all_dialog_lens = []
        for raw_dialog, laed_z in zip(data, laed_z_data):
            domain = raw_dialog['scenario']['task']['intent']
            kb_items = []
            if raw_dialog['scenario']['kb']['items'] is not None:
                for item in raw_dialog['scenario']['kb']['items']:
                    kb_items.append([KB]+self.tokenize(" ".join(["{} {}".format(k, v) for k, v in item.items()])))

            dialog = [Pack(utt=[BOS, domain, BOD, EOS], speaker=USR, slots=None, domain=domain)]
            for turn, laed_z_turn in zip(raw_dialog['dialogue'], laed_z):
                utt = turn['data']['utterance']
                slots = turn['data'].get('slots')
                speaker = self.speaker_map[turn['turn']]
                if self.config.include_domain:
                    utt = [BOS, speaker, domain] + self.tokenize(utt) + [EOS]
                else:
                    utt = [BOS, speaker] + self.tokenize(utt) + [EOS]

                all_lens.append(len(utt))
                if speaker == SYS:
                    dialog.append(Pack(utt=utt, speaker=speaker, slots=slots, domain=domain, kb=kb_items))
                else:
                    dialog.append(Pack(utt=utt, speaker=speaker, slots=slots, domain=domain, kb=[]))
                self._process_domain_description(turn, domain, laed_z_turn)

            all_dialog_lens.append(len(dialog))
            new_dialog.append(dialog)

        print("Max utt len %d, mean utt len %.2f" % (
            np.max(all_lens), float(np.mean(all_lens))))
        print("Max dialog len %d, mean dialog len %.2f" % (
            np.max(all_dialog_lens), float(np.mean(all_dialog_lens))))
        return new_dialog

    def _process_domain_description(self, in_turn, in_domain, in_laed_z):
        utt = in_turn['data']['utterance']
        speaker = self.speaker_map[in_turn['turn']]

        action = []
        nlu_dict = in_turn['data'].get('nlu', {}).get('annotations', {})
        if 'ner' in nlu_dict:
            action += flatten_nlu_dict(nlu_dict['ner'])
        if 'entity_linking' in nlu_dict:
            for key, value in nlu_dict['entity_linking'].items():
                actual_value = value
                if type(value) == list:
                    actual_value = value[0]
                action.append(actual_value['entity'])
        if 'SUTime' in nlu_dict:
            for item in nlu_dict['SUTime']:
                action.append(item['text'])
        action = ' '.join(action)
        empty_action = not len(action.strip())
        if self.config.include_domain:
            utt = [BOS, speaker, in_domain] + self.tokenize(utt) + [EOS]
            action = [BOS, speaker, in_domain] + self.tokenize(action) + [EOS]
        else:
            utt = [BOS, speaker] + self.tokenize(utt) + [EOS]
            action = [BOS, speaker] + self.tokenize(action) + [EOS]
        self.domain_descriptions.append(Pack(domain=in_domain,
                                             speaker=speaker,
                                             utt=utt,
                                             actions=action,
                                             laed_z=in_laed_z,
                                             empty_action=empty_action))

    def _build_vocab(self):
        all_words = []
        for dialog in self.train_corpus:
            for turn in dialog:
                all_words.extend(turn.utt)
                for item in turn.get('kb', []):
                    all_words.extend(item)

        for resp in self.domain_descriptions:
            all_words.extend(resp.actions)

        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count])

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.valid_corpus),
                 len(self.test_corpus),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1],
                 float(discard_wc) / len(all_words)))

        self.vocab = [PAD, UNK, SYS, USR] + [t for t, cnt in vocab_count]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab[UNK]

    def _load_laed_z(self):
        laed_z = defaultdict(lambda: [])

        for folder in self.config.laed_z_folders:  
            laed_z_part = load_laed_features(folder)
            for dataset_name in laed_z_part['dialog']:
                laed_z[dataset_name].append(laed_z_part['dialog'][dataset_name])
        self.laed_z = {}
        for dataset_name in ['train', 'valid', 'test']:
            self.laed_z[dataset_name] = self._combine_laed_z_features(laed_z[dataset_name])
        self.laed_z_size = self.laed_z['train'][0][0].shape[-1]

    def _combine_laed_z_features(self, in_features):
        result = []
        for i in range(len(in_features[0])):
            result.append([])
            for j in range(len(in_features[0][i])):
                features_i_j = [feature[i][j] for feature in in_features]
                result[-1].append(np.concatenate(features_i_j, axis=-1))
        return result

    def _sent2id(self, sent):
        return [self.rev_vocab.get(t, self.unk_id) for t in sent]

    def _to_id_corpus(self, name, data, laed_z, use_black_list, domains):
        results = []
        kick_cnt = 0
        domain_cnt = []
        assert len(data) == len(laed_z), 'Lengths of dialog dataset and laed_z do not match'
        for dialog, dialog_laed_z in zip(data, laed_z):
            if len(dialog) < 1:
                continue
            domain = dialog[0].domain
            if domain not in domains:
                continue
            should_filter = np.random.rand() < self.black_ratio
            if use_black_list and self.black_domains \
                    and domain in self.black_domains \
                    and should_filter:
                kick_cnt += 1
                continue
            temp = []
            # convert utterance and feature into numeric numbers
            for turn, laed_z_turn in zip(dialog, dialog_laed_z):
                id_turn = Pack(utt=self._sent2id(turn.utt),
                               utt_raw=turn.utt,
                               speaker=turn.speaker,
                               domain=turn.domain,
                               domain_id=self.rev_vocab[domain],
                               meta=turn.get('meta'),
                               kb=[self._sent2id(item) for item in turn.get('kb', [])],
                               kb_raw=turn.get('kb', []),
                               laed_z=laed_z_turn)
                temp.append(id_turn)

            results.append(temp)
            domain_cnt.append(domain)
        self.logger.info("Filter {} samples from {}".format(kick_cnt, name))
        self.logger.info(Counter(domain_cnt).most_common())
        return results

    def get_corpus(self, domains=None):
        real_domains = domains if domains else self.domains
        id_train = self._to_id_corpus("Train",
                                      self.train_corpus,
                                      self.laed_z['train'],
                                      use_black_list=True,
                                      domains=real_domains)
        id_valid = self._to_id_corpus("Valid",
                                      self.valid_corpus,
                                      self.laed_z['valid'],
                                      use_black_list=False,
                                      domains=real_domains)
        id_test = self._to_id_corpus("Test",
                                     self.test_corpus,
                                     self.laed_z['test'],
                                     use_black_list=False,
                                     domains=real_domains)
        return Pack(train=id_train, valid=id_valid, test=id_test)

    def get_seed_responses(self, utt_cnt=defaultdict(lambda: 100)):
        domain_seeds = defaultdict(list)
        all_domains = []
        if sum(utt_cnt.values()) == 0 or self.config.action_match is False:
            return []

        for resp in self.domain_descriptions:
            if resp.empty_action:
                continue
            resp_copy = resp.copy()
            resp_copy['utt'] = self._sent2id(resp.utt)
            resp_copy['actions'] = self._sent2id(resp.actions)
            resp_copy['domain_id'] = self.rev_vocab[resp.domain]
            if len(domain_seeds[resp.domain]) >= utt_cnt[resp.domain]:
                continue

            domain_seeds[resp.domain].append(resp_copy)
            all_domains.append(resp.domain)

        seed_responses = []
        for v in domain_seeds.values():
            seed_responses.extend(v)

        self.logger.info("Collected {} extra samples".format(len(seed_responses)))
        self.logger.info(Counter(all_domains).most_common())
        return seed_responses

