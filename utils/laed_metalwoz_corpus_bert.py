import logging

SEP = '[SEP]'
CLS = '[CLS]'
UNK = '[UNK]'
SYS = 'system'
USR = 'user'


class LAEDMetaLWOzCorpusBERT(object):
    logger = logging.getLogger()

    def __init__(self, config):
        self.config = config
        self._path = config.data_dir[0]
        self.max_utt_len = config.max_utt_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenize = self.tokenizer.tokenize
        self.corpus = self._read_file(self._path)
        self.train_corpus, devtest = train_test_split(self.corpus, test_size=0.2, random_state=271)
        self.valid_corpus, self.test_corpus = train_test_split(devtest, test_size=0.5, random_state=271)
        self.vocab = self.tokenizer.vocab
        self.sep_id, self.cls_id = self.vocab[SEP], self.vocab[CLS]
        self.unk_id = self.vocab[UNK]
        print("Done loading corpus")

    def _read_file(self, path):
        with open(path) as json_in:
            data = json.load(json_in)
        return self._process_dialogs(data, exclude_domains=self.config.exclude_domains)

    def _process_dialogs(self, data, exclude_domains=[]):
        new_dialog = []
        bod_utt = [SEP]
        eod_utt = [SEP]
        all_lens = []
        all_dialog_lens = []
        speaker_map = {'bot': SYS, 'user': USR}
        for raw_dialog in data:
            domain = raw_dialog['task']['domain']
            if domain in exclude_domains:
                continue
            dialog = [Pack(utt=bod_utt, speaker=0, meta=None)]

            for turn in raw_dialog['turns']:
                utt = turn['text']
                utt = [speaker_map[turn['authorType']]] + self.tokenize(utt)
                all_lens.append(len(utt))
                dialog.append(Pack(utt=utt, speaker=turn['authorType']))

            if not hasattr(self.config, 'include_eod') or self.config.include_eod:
                dialog.append(Pack(utt=eod_utt, speaker=0))

            all_dialog_lens.append(len(dialog))
            new_dialog.append(dialog)

        print("Max utt len %d, mean utt len %.2f" % (np.max(all_lens), float(np.mean(all_lens))))
        print("Max dialog len %d, mean dialog len %.2f" % (np.max(all_dialog_lens),
                                                           float(np.mean(all_dialog_lens))))
        return new_dialog

    def _sent2id(self, sent):
        return self.tokenizer.convert_tokens_to_ids(sent)

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
