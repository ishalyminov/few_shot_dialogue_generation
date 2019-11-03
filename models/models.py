import numpy as np

import torch
import torch.nn as nn

from zsdg.dataset.corpora import PAD, BOS, EOS, BOD
from zsdg import criterions
from zsdg.enc2dec.encoders import EncoderRNN, RnnUttEncoder
from zsdg.utils import INT, FLOAT, LONG, cast_type, Pack
from zsdg import nn_lib
from zsdg.nn_lib import IdentityConnector, Bi2UniConnector
from zsdg.enc2dec.decoders import GEN
from zsdg.models.models import PtrBase
from zsdg.models.model_bases import BaseModel

from enc2dec.elmo import ElmoEmbedding
from enc2dec.encoders import ElmoUttEncoder
from zsdg.enc2dec.decoders import DecoderPointerGen, DecoderRNN

class ElmoPtrBase(PtrBase):
    def __init__(self, config, corpus):
        super(ElmoPtrBase, self).__init__(config)

        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.pad_id = self.rev_vocab[PAD]

        self.embedding = ElmoEmbedding(config)
        self.utt_encoder = ElmoUttEncoder(config,
                                          config.utt_cell_size,
                                          config.dropout,
                                          use_attn=config.utt_type == 'rnn_attn',
                                          vocab_size=self.vocab_size,
                                          embedding=self.embedding,
                                          feat_size=1)


class HRED(BaseModel):
    def valid_loss(self, loss, batch_cnt=None):
        return loss.nll

    def __init__(self, corpus, config):
        super(HRED, self).__init__(config)

        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.pad_id = self.rev_vocab[PAD]

        # build model here
        self.utt_encoder = ElmoUttEncoder(config, config.utt_cell_size, config.dropout,
                                          use_attn=config.utt_type == 'attn_rnn',
                                          feat_size=1)

        self.ctx_encoder = EncoderRNN(self.utt_encoder.output_size,
                                      config.ctx_cell_size,
                                      0.0,
                                      config.dropout,
                                      config.num_layer,
                                      config.rnn_cell,
                                      variable_lengths=False,
                                      bidirection=config.bi_ctx_cell)

        if config.bi_ctx_cell or config.num_layer > 1:
            self.connector = Bi2UniConnector(config.rnn_cell, config.num_layer,
                                             config.ctx_cell_size,
                                             config.dec_cell_size)
        else:
            self.connector = IdentityConnector()

        self.decoder = ElmoDecoderRNN(self.vocab_size, config.max_dec_len,
                                  self.utt_encoder.embed_size, config.dec_cell_size,
                                  self.go_id, self.eos_id,
                                  self.vocab,
                                  n_layers=1, rnn_cell=config.rnn_cell,
                                  input_dropout_p=config.dropout,
                                  dropout_p=config.dropout,
                                  use_attention=config.use_attn,
                                  embedding=self.utt_encoder.embedding,
                                  attn_size=self.ctx_encoder.output_size,
                                  attn_mode=config.attn_type,
                                  use_gpu=config.use_gpu)
        self.nll = criterions.NLLEntropy(self.pad_id, config)

    def forward(self, data_feed, mode, gen_type='greedy', return_latent=False):
        """
        B: batch_size, D: context_size U: utt_size, X: response_size
        1. ctx_lens: B x 1
        2. ctx_utts: B x D x U
        3. ctx_confs: B x D
        4. ctx_floors: B x D
        5. out_lens: B x 1
        6. out_utts: B x X

        :param data_feed:
        {'ctx_lens': vec_ctx_lens, 'ctx_utts': vec_ctx_utts,
         'ctx_confs': vec_ctx_confs, 'ctx_floors': vec_ctx_floors,
         'out_lens': vec_out_lens, 'out_utts': vec_out_utts}
        :param return_label
        :param dec_type
        :return: outputs
        """
        ctx_lens = data_feed['context_lens']
        ctx_utts = self.np2var(data_feed['contexts'], LONG)
        ctx_confs = self.np2var(data_feed['context_confs'], FLOAT)
        batch_size = len(ctx_lens)

        ctx_utts_elmo = self.np2var(data_feed.get('contexts_elmo'), LONG)

        # required fields
        out_utts = self.np2var(data_feed['outputs'], LONG)
        out_utts_elmo = self.np2var(data_feed['outputs_elmo'], LONG)

        enc_inputs = self.utt_encoder(ctx_utts_elmo, ctx_confs)

        enc_outs, enc_last = self.ctx_encoder(enc_inputs, ctx_lens)

        # get decoder inputs
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts_elmo[:, 0:-1, :]

        # pack attention context
        if self.config.use_attn:
            attn_inputs = enc_outs
        else:
            attn_inputs = None

        # create decoder initial states
        dec_init_state = self.connector(enc_last)

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size,
                                                   dec_inputs,
                                                   dec_init_state,
                                                   attn_context=attn_inputs,
                                                   mode=mode,
                                                   gen_type=gen_type,
                                                   beam_size=self.config.beam_size)
        if mode == GEN:
            return dec_ctx, labels
        else:
            if return_latent:
                return Pack(nll=self.nll(dec_outs, labels), latent_actions=dec_init_state)
            else:
                return Pack(nll=self.nll(dec_outs, labels))


class ZeroShotLAPtrHRED(PtrBase):
    def __init__(self, corpus, config):
        super(ZeroShotLAPtrHRED, self).__init__(config)

        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.pad_id = self.rev_vocab[PAD]

        # build model here
        self.embedding = nn.Embedding(self.vocab_size, config.embed_size, padding_idx=self.pad_id)
        self.utt_encoder = RnnUttEncoder(config.utt_cell_size,
                                         config.dropout,
                                         use_attn=config.utt_type == 'rnn_attn',
                                         vocab_size=self.vocab_size,
                                         embedding=self.embedding,
                                         feat_size=1)
        self.ctx_encoder = EncoderRNN(self.utt_encoder.output_size,
                                      config.ctx_cell_size,
                                      0.0,
                                      config.dropout,
                                      config.num_layer,
                                      config.rnn_cell,
                                      variable_lengths=False,
                                      bidirection=config.bi_ctx_cell)
        self.policy = nn.Linear(self.ctx_encoder.output_size, config.dec_cell_size)
        self.utt_policy = lambda x: x
        combined_dec_init_state_size = config.dec_cell_size + corpus.laed_z_size
        self.connector = nn_lib.LinearConnector(combined_dec_init_state_size,
                                                config.dec_cell_size,
                                                is_lstm=config.rnn_cell == 'lstm')
        self.attn_size = self.ctx_encoder.output_size
        self.decoder = DecoderPointerGen(self.vocab_size,
                                         config.max_dec_len,
                                         config.embed_size,
                                         config.dec_cell_size,
                                         self.go_id,
                                         self.eos_id,
                                         n_layers=1,
                                         rnn_cell=config.rnn_cell,
                                         input_dropout_p=config.dropout,
                                         dropout_p=config.dropout,
                                         attn_size=self.attn_size,
                                         attn_mode=config.attn_type,
                                         use_gpu=config.use_gpu,
                                         embedding=self.embedding)

        self.nll_loss = criterions.NLLEntropy(self.pad_id, config)
        self.l2_loss = criterions.L2Loss()

    def valid_loss(self, loss, batch_cnt=None):
        total_loss = loss.distance + loss.nll + 0.01 * loss.attn_loss
        return total_loss

    def forward(self, data_feed, mode, gen_type='greedy', return_latent=False):
        # optional fields
        ctx_lens = data_feed.get('context_lens')
        ctx_utts = self.np2var(data_feed.get('contexts'), LONG)
        ctx_confs = self.np2var(data_feed.get('context_confs'), FLOAT)
        out_acts = self.np2var(data_feed.get('output_actions'), LONG)

        # required fields
        out_utts = self.np2var(data_feed['outputs'], LONG)
        batch_size = len(data_feed['outputs'])
        out_confs = self.np2var(np.ones((batch_size, 1)), FLOAT)
        laed_z = self.np2var(data_feed.get('laed_z'), FLOAT)

        out_embedded, out_outs, _, _ = self.utt_encoder(out_utts.unsqueeze(1),
                                                        out_confs,
                                                        return_all=True)
        out_embedded = self.utt_policy(out_embedded.squeeze(1))

        if ctx_lens is None:
            act_embedded, act_outs, _, _ = self.utt_encoder(out_acts.unsqueeze(1),
                                                            out_confs,
                                                            return_all=True)
            act_embedded = act_embedded.squeeze(1)

            # create attention contexts
            attn_inputs = act_outs.contiguous().view(batch_size, -1, self.utt_encoder.output_size)
            attn_words = out_acts.view(batch_size, -1)
            latent_action = self.utt_policy(act_embedded)
        else:
            utt_embedded, utt_outs, _, _ = self.utt_encoder(ctx_utts, ctx_confs, return_all=True)
            ctx_outs, ctx_last = self.ctx_encoder(utt_embedded, ctx_lens)
            pi_inputs = self._gather_last_out(ctx_outs, ctx_lens)

            # create decoder initial states
            latent_action = self.policy(pi_inputs)

            # create attention contexts
            ctx_outs = ctx_outs.unsqueeze(2)\
                               .repeat(1, 1, ctx_utts.size(2), 1)\
                               .view(batch_size, -1, self.ctx_encoder.output_size)
            utt_outs = utt_outs.contiguous().view(batch_size, -1, self.utt_encoder.output_size)
            attn_inputs = ctx_outs + utt_outs  # batch_size x num_word x attn_size
            attn_words = ctx_utts.view(batch_size, -1)  # batch_size x num_words

        latent_state_combined = [latent_action] + [laed_z]

        dec_init_state = self.connector(torch.cat(latent_state_combined, dim=-1))

        # mask out PAD words in the attention inputs
        attn_inputs, attn_words = self._remove_padding(attn_inputs, attn_words)

        # get decoder inputs
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size,
                                                   attn_inputs,
                                                   attn_words,
                                                   inputs=dec_inputs,
                                                   init_state=dec_init_state,
                                                   mode=mode,
                                                   gen_type=gen_type)
        if mode == GEN:
            return dec_ctx, labels
        else:
            loss_pack = self.compute_loss(dec_outs, dec_ctx, labels)
            if return_latent:
                loss_pack['latent_actions'] = latent_action
            loss_pack['distance'] = self.l2_loss(out_embedded, latent_action)
            return loss_pack


class LAPtrHRED(ElmoPtrBase):
    def valid_loss(self, loss, batch_cnt=None):
        total_loss = loss.nll + 0.01 * loss.attn_loss
        return total_loss

    def __init__(self, corpus, config):
        super(LAPtrHRED, self).__init__(config, corpus)

        self.ctx_encoder = EncoderRNN(self.utt_encoder.output_size,
                                      config.ctx_cell_size,
                                      0.0,
                                      config.dropout,
                                      config.num_layer,
                                      config.rnn_cell,
                                      variable_lengths=False,
                                      bidirection=config.bi_ctx_cell)

        combined_dec_init_state_size = config.dec_cell_size + corpus.laed_z_size
        if config.bi_ctx_cell or config.num_layer > 1:
            self.connector = Bi2UniConnector(config.rnn_cell,
                                             config.num_layer,
                                             combined_dec_init_state_size,
                                             config.dec_cell_size)
        else:
            self.connector = nn.Linear(combined_dec_init_state_size, config.dec_cell_size)

        self.attn_size = self.ctx_encoder.output_size

        self.plain_embedding = nn.Embedding(self.vocab_size, config.embed_size)
        self.decoder = DecoderPointerGen(self.vocab_size,
                                         config.max_dec_len,
                                         config.embed_size,
                                         config.dec_cell_size,
                                         self.go_id,
                                         self.eos_id,
                                         n_layers=1,
                                         rnn_cell=config.rnn_cell,
                                         input_dropout_p=config.dropout,
                                         dropout_p=config.dropout,
                                         attn_size=self.attn_size,
                                         attn_mode=config.attn_type,
                                         use_gpu=config.use_gpu,
                                         embedding=self.plain_embedding)

        self.nll_loss = criterions.NLLEntropy(self.pad_id, config)

    def forward(self, data_feed, mode, gen_type='greedy', return_latent=False):
        """
        B: batch_size, D: context_size U: utt_size, X: response_size
        1. ctx_lens: B x 1
        2. ctx_utts: B x D x U
        3. ctx_confs: B x D
        4. ctx_floors: B x D
        5. out_lens: B x 1
        6. out_utts: B x X

        :param data_feed:
        {'ctx_lens': vec_ctx_lens, 'ctx_utts': vec_ctx_utts,
         'ctx_confs': vec_ctx_confs, 'ctx_floors': vec_ctx_floors,
         'out_lens': vec_out_lens, 'out_utts': vec_out_utts}
        :param return_label
        :param dec_type
        :return: outputs
        """
        ctx_lens = data_feed['context_lens']
        ctx_utts = self.np2var(data_feed['contexts'], LONG)
        ctx_confs = self.np2var(data_feed['context_confs'], FLOAT)
        out_utts = self.np2var(data_feed['outputs'], LONG)
        laed_z = self.np2var(data_feed.get('laed_z'), FLOAT)
        batch_size = len(ctx_lens)

        ctx_utts_elmo = self.np2var(data_feed.get('contexts_elmo'), LONG)
        utt_embedded, utt_outs, _, _ = self.utt_encoder(ctx_utts_elmo, ctx_confs, return_all=True)

        ctx_outs, ctx_last = self.ctx_encoder(utt_embedded, ctx_lens)

        # get decoder inputs
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        # latent_state_combined = [ctx_last] + [laed_z]

        # create decoder initial states
        dec_init_state = [self.connector(torch.cat([ctx_last_i, laed_z.unsqueeze(0)], dim=-1))
                          for ctx_last_i in list(ctx_last)]
        # dec_init_state = self.connector(ctx_last)

        # attention
        ctx_outs = ctx_outs\
            .unsqueeze(2)\
            .repeat(1, 1, ctx_utts.size(2), 1)\
            .view(batch_size, -1, self.ctx_encoder.output_size)
        utt_outs = utt_outs.contiguous().view(batch_size, -1, self.utt_encoder.output_size)
        attn_inputs = ctx_outs + utt_outs
        flat_ctx_words = ctx_utts.view(batch_size, -1)

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size,
                                                   attn_inputs,
                                                   flat_ctx_words,
                                                   inputs=dec_inputs,
                                                   init_state=dec_init_state,
                                                   mode=mode,
                                                   gen_type=gen_type)
        if mode == GEN:
            return dec_ctx, labels
        else:
            results = self.compute_loss(dec_outs, dec_ctx, labels)
            if return_latent:
                results['latent_actions'] = dec_init_state
            return results


class PtrHRED(ElmoPtrBase):

    def valid_loss(self, loss, batch_cnt=None):
        total_loss = loss.nll + 0.01 * loss.attn_loss
        return total_loss

    def __init__(self, corpus, config):
        super(PtrHRED, self).__init__(config, corpus)

        self.ctx_encoder = EncoderRNN(self.utt_encoder.output_size,
                                      config.ctx_cell_size,
                                      0.0,
                                      config.dropout,
                                      config.num_layer,
                                      config.rnn_cell,
                                      variable_lengths=False,
                                      bidirection=config.bi_ctx_cell)

        if config.bi_ctx_cell or config.num_layer > 1:
            self.connector = Bi2UniConnector(config.rnn_cell, config.num_layer,
                                             config.ctx_cell_size,
                                             config.dec_cell_size)
        else:
            self.connector = IdentityConnector()

        self.attn_size = self.ctx_encoder.output_size

        self.plain_embedding = nn.Embedding(self.vocab_size, config.embed_size)
        self.decoder = DecoderPointerGen(self.vocab_size,
                                         config.max_dec_len,
                                         config.embed_size,
                                         config.dec_cell_size,
                                         self.go_id,
                                         self.eos_id,
                                         n_layers=1,
                                         rnn_cell=config.rnn_cell,
                                         input_dropout_p=config.dropout,
                                         dropout_p=config.dropout,
                                         attn_size=self.attn_size,
                                         attn_mode=config.attn_type,
                                         use_gpu=config.use_gpu,
                                         embedding=self.plain_embedding)

        self.nll_loss = criterions.NLLEntropy(self.pad_id, config)

    def forward(self, data_feed, mode, gen_type='greedy', return_latent=False):
        """
        B: batch_size, D: context_size U: utt_size, X: response_size
        1. ctx_lens: B x 1
        2. ctx_utts: B x D x U
        3. ctx_confs: B x D
        4. ctx_floors: B x D
        5. out_lens: B x 1
        6. out_utts: B x X

        :param data_feed:
        {'ctx_lens': vec_ctx_lens, 'ctx_utts': vec_ctx_utts,
         'ctx_confs': vec_ctx_confs, 'ctx_floors': vec_ctx_floors,
         'out_lens': vec_out_lens, 'out_utts': vec_out_utts}
        :param return_label
        :param dec_type
        :return: outputs
        """
        ctx_lens = data_feed['context_lens']
        ctx_utts = self.np2var(data_feed['contexts'], LONG)
        ctx_confs = self.np2var(data_feed['context_confs'], FLOAT)
        batch_size = len(ctx_lens)

        ctx_utts_elmo = self.np2var(data_feed.get('contexts_elmo'), LONG)

        # required fields
        out_utts = self.np2var(data_feed['outputs'], LONG)
        # out_utts_elmo = self.np2var(data_feed['outputs_elmo'], LONG)
        out_confs = self.np2var(np.ones((batch_size, 1)), FLOAT)

        utt_embedded, utt_outs, _, _ = self.utt_encoder(ctx_utts_elmo, ctx_confs, return_all=True)

        ctx_outs, ctx_last = self.ctx_encoder(utt_embedded, ctx_lens)

        # get decoder inputs
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        # create decoder initial states
        dec_init_state = self.connector(ctx_last)

        # attention
        ctx_outs = ctx_outs.unsqueeze(2).repeat(1, 1, ctx_utts.size(2), 1).view(batch_size, -1, self.ctx_encoder.output_size)
        utt_outs = utt_outs.contiguous().view(batch_size, -1, self.utt_encoder.output_size)
        attn_inputs = ctx_outs + utt_outs
        flat_ctx_words = ctx_utts.view(batch_size, -1)

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size,
                                                   attn_inputs,
                                                   flat_ctx_words,
                                                   inputs=dec_inputs,
                                                   init_state=dec_init_state,
                                                   mode=mode,
                                                   gen_type=gen_type)
        if mode == GEN:
            return dec_ctx, labels
        else:
            results = self.compute_loss(dec_outs, dec_ctx, labels)
            if return_latent:
                results['latent_actions'] = dec_init_state
            return results


class ZeroShotPtrHRED(ElmoPtrBase):
    def __init__(self, corpus, config):
        super(ZeroShotPtrHRED, self).__init__(config, corpus)

        self.ctx_encoder = EncoderRNN(self.utt_encoder.output_size,
                                      config.ctx_cell_size,
                                      0.0,
                                      config.dropout,
                                      config.num_layer,
                                      config.rnn_cell,
                                      variable_lengths=False,
                                      bidirection=config.bi_ctx_cell)

        self.policy = nn.Linear(self.ctx_encoder.output_size, config.dec_cell_size)
        self.utt_policy = lambda x: x

        self.connector = nn_lib.LinearConnector(config.dec_cell_size,
                                                config.dec_cell_size,
                                                is_lstm=config.rnn_cell == 'lstm')

        self.attn_size = self.ctx_encoder.output_size
        self.decoder = DecoderPointerGen(self.vocab_size,
                                         config.max_dec_len,
                                         self.embedding.embedding_dim,
                                         config.dec_cell_size,
                                         self.go_id,
                                         self.eos_id,
                                         n_layers=1,
                                         rnn_cell=config.rnn_cell,
                                         input_dropout_p=config.dropout,
                                         dropout_p=config.dropout,
                                         attn_size=self.attn_size,
                                         attn_mode=config.attn_type,
                                         use_gpu=config.use_gpu,
                                         embedding=self.embedding)

        self.nll_loss = criterions.NLLEntropy(self.pad_id, config)
        self.l2_loss = criterions.L2Loss()

    def valid_loss(self, loss, batch_cnt=None):
        total_loss = loss.distance + loss.nll + 0.01 * loss.attn_loss
        return total_loss

    def forward(self, data_feed, mode, gen_type='greedy', return_latent=False):

        # optional fields
        ctx_lens = data_feed.get('context_lens')
        ctx_utts = self.np2var(data_feed.get('contexts'), LONG)
        ctx_confs = self.np2var(data_feed.get('context_confs'), FLOAT)
        out_acts = self.np2var(data_feed.get('output_actions'), LONG)

        ctx_utts_elmo = self.np2var(data_feed.get('contexts_elmo'), LONG)

        # required fields
        out_utts = self.np2var(data_feed['outputs'], LONG)
        out_utts_elmo = self.np2var(data_feed['outputs_elmo'], LONG)
        batch_size = len(data_feed['outputs'])
        out_confs = self.np2var(np.ones((batch_size, 1)), FLOAT)

        out_embedded, out_outs, _, _ = self.utt_encoder(out_utts_elmo.unsqueeze(1), out_confs, return_all=True)
        out_embedded = self.utt_policy(out_embedded.squeeze(1))

        if ctx_lens is None:
            act_embedded, act_outs, _, _ = self.utt_encoder(out_acts.unsqueeze(1), out_confs, return_all=True)
            act_embedded = act_embedded.squeeze(1)

            # create attention contexts
            attn_inputs = act_outs.contiguous().view(batch_size, -1, self.utt_encoder.output_size)
            attn_words = out_acts.view(batch_size, -1)
            latent_action = self.utt_policy(act_embedded)
        else:
            utt_embedded, utt_outs, _, _ = self.utt_encoder(ctx_utts_elmo, ctx_confs, return_all=True)
            ctx_outs, ctx_last = self.ctx_encoder(utt_embedded, ctx_lens)
            pi_inputs = self._gather_last_out(ctx_outs, ctx_lens)

            # create decoder initial states
            latent_action = self.policy(pi_inputs)

            # create attention contexts
            ctx_outs = ctx_outs.unsqueeze(2).repeat(1, 1, ctx_utts.size(2), 1).view(batch_size, -1, self.ctx_encoder.output_size)
            utt_outs = utt_outs.contiguous().view(batch_size, -1, self.utt_encoder.output_size)
            attn_inputs = ctx_outs + utt_outs  # batch_size x num_word x attn_size
            attn_words = ctx_utts.view(batch_size, -1)  # batch_size x num_words

        dec_init_state = self.connector(latent_action)

        # mask out PAD words in the attention inputs
        attn_inputs, attn_words = self._remove_padding(attn_inputs, attn_words)

        # get decoder inputs
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts_elmo[:, 0:-1, :]

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size,
                                                   attn_inputs,
                                                   attn_words,
                                                   inputs=dec_inputs,
                                                   init_state=dec_init_state,
                                                   mode=mode,
                                                   gen_type=gen_type)
        if mode == GEN:
            return dec_ctx, labels
        else:
            loss_pack = self.compute_loss(dec_outs, dec_ctx, labels)
            if return_latent:
                loss_pack['latent_actions'] = latent_action
            loss_pack['distance'] = self.l2_loss(out_embedded, latent_action)
            return loss_pack
