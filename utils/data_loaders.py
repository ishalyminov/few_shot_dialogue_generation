from __future__ import print_function
import numpy as np
from zsdg.utils import Pack
from zsdg.dataset.corpora import SimDialCorpus, SYS, USR
from zsdg.dataset.dataloader_bases import DataLoader, LongDataLoader

from zsdg.dataset.data_loaders import *


# Maluuba DI-VAE
class MaluubaDataLoader(DataLoader):
    def __init__(self, name, data, config):
        super(MaluubaDataLoader, self).__init__(name, fix_batch=config.fix_batch)
        self.name = name
        self.max_utt_size = config.max_utt_len
        self.data = self.flatten_dialog(data, config.backward_size)
        self.data_size = len(self.data)
        if config.fix_batch:
            all_ctx_lens = [len(d.context) for d in self.data]
            self.indexes = list(np.argsort(all_ctx_lens))[::-1]
        else:
            self.indexes = range(len(self.data))

    def flatten_dialog(self, data, backward_size):
        results = []
        for dialog in data:
            for i in range(1, len(dialog)):
                e_id = i
                s_id = max(0, e_id - backward_size)
                response = dialog[i].copy()

                if response.speaker == USR:
                    continue

                response['utt'] = self.pad_to(self.max_utt_size, response.utt, do_pad=False)
                contexts = []
                for turn in dialog[s_id:e_id]:
                    turn['utt'] = self.pad_to(self.max_utt_size, turn.utt, do_pad=False)
                    contexts.append(turn)
                results.append(Pack(context=contexts, response=response))
        return results

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]
        # input_context, context_lens, floors, topics, a_profiles, b_Profiles, outputs, output_lens
        context_lens, context_utts, out_utts, out_lens = [], [], [], []
        metas = []
        for row in rows:
            ctx = row.context
            resp = row.response

            out_utt = resp.utt
            context_lens.append(len(ctx))
            context_utts.append([turn.utt for turn in ctx])

            out_utt = out_utt
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))
            metas.append(resp.meta)

        vec_context_lens = np.array(context_lens)
        vec_context = np.zeros((self.batch_size, np.max(vec_context_lens),
                                self.max_utt_size), dtype=np.int32)
        vec_outs = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_out_lens = np.array(out_lens)

        for b_id in range(self.batch_size):
            vec_outs[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            # fill the context tensor
            new_array = np.empty((vec_context_lens[b_id], self.max_utt_size))
            new_array.fill(0)
            for i, row in enumerate(context_utts[b_id]):
                for j, ele in enumerate(row):
                    new_array[i, j] = ele
            vec_context[b_id, 0:vec_context_lens[b_id], :] = new_array

        return Pack(contexts=vec_context,
                    context_lens=vec_context_lens,
                    outputs=vec_outs,
                    output_lens=vec_out_lens,
                    metas=metas)


# Maluuba DI-VST
class MaluubaDialogSkipLoader(DataLoader):
    def __init__(self, name, data, config):
        super(MaluubaDialogSkipLoader, self).__init__(name, fix_batch=config.fix_batch)
        self.name = name
        self.max_utt_size = config.max_utt_len
        self.data = self.flatten_dialog(data, config.backward_size)
        self.data_size = len(self.data)
        if config.fix_batch:
            all_ctx_lens = [len(d.context) for d in self.data]
            self.indexes = list(np.argsort(all_ctx_lens))[::-1]
        else:
            self.indexes = range(len(self.data))

    def flatten_dialog(self, data, backward_size):
        results = []
        for dialog in data:
            for i in range(1, len(dialog)-1):
                e_id = i
                s_id = max(0, e_id - backward_size)

                response = dialog[i]
                if response.speaker == USR:
                    continue

                prev = dialog[i - 1]
                next = dialog[i + 1]

                response['utt'] = self.pad_to(self.max_utt_size, response.utt, do_pad=False)
                prev['utt'] = self.pad_to(self.max_utt_size, prev.utt, do_pad=False)
                next['utt'] = self.pad_to(self.max_utt_size, next.utt, do_pad=False)

                contexts = []
                for turn in dialog[s_id:e_id]:
                    turn['utt'] = self.pad_to(self.max_utt_size, turn.utt, do_pad=False)
                    contexts.append(turn)

                results.append(Pack(context=contexts,
                                    response=response,
                                    prev_resp=prev,
                                    next_resp=next))
        return results

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]

        context_lens, context_utts, out_utts, out_lens = [], [], [], []
        prev_utts, prev_lens = [], []
        next_utts, next_lens = [], []
        metas = []
        for row in rows:
            ctx = row.context
            resp = row.response

            out_utt = resp.utt
            context_lens.append(len(ctx))
            context_utts.append([turn.utt for turn in ctx])

            out_utt = out_utt
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))
            metas.append(resp.meta)

            prev_utts.append(row.prev_resp.utt)
            prev_lens.append(len(row.prev_resp.utt))

            next_utts.append(row.next_resp.utt)
            next_lens.append(len(row.next_resp.utt))

        vec_context_lens = np.array(context_lens)
        vec_context = np.zeros((self.batch_size, np.max(vec_context_lens),
                                self.max_utt_size), dtype=np.int32)
        vec_outs = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_prevs = np.zeros((self.batch_size, np.max(prev_lens)), dtype=np.int32)
        vec_nexts = np.zeros((self.batch_size, np.max(next_lens)), dtype=np.int32)
        vec_out_lens = np.array(out_lens)
        vec_prev_lens = np.array(prev_lens)
        vec_next_lens = np.array(next_lens)

        for b_id in range(self.batch_size):
            vec_outs[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            vec_prevs[b_id, 0:vec_prev_lens[b_id]] = prev_utts[b_id]
            vec_nexts[b_id, 0:vec_next_lens[b_id]] = next_utts[b_id]

            # fill the context tensor
            new_array = np.empty((vec_context_lens[b_id], self.max_utt_size))
            new_array.fill(0)
            for i, row in enumerate(context_utts[b_id]):
                for j, ele in enumerate(row):
                    new_array[i, j] = ele
            vec_context[b_id, 0:vec_context_lens[b_id], :] = new_array

        return Pack(contexts=vec_context,
                    context_lens=vec_context_lens,
                    outputs=vec_outs,
                    output_lens=vec_out_lens,
                    metas=metas,
                    prevs=vec_prevs,
                    prev_lens=vec_prev_lens,
                    nexts=vec_nexts,
                    next_lens=vec_next_lens)


# Maluuba ZSL
class ZslMaluubaDataLoader(DataLoader):
    def __init__(self, name, data, laed_z_dialog, laed_z_seed, config, warmup_data=None):
        super(ZslMaluubaDataLoader, self).__init__(name)
        self.max_utt_size = config.max_utt_len

        self.data, self.laed_z_dialog = self.flatten_dialog(data, laed_z_dialog, config.backward_size)
        self.data_size = len(self.data)
        data_lens = [len(line.context) for line in self.data]
        if False:
            self.indexes = list(np.argsort(data_lens))[::-1]
        else:
            self.indexes = range(len(data_lens))

        assert self.laed_z_dialog.shape[0] == len(self.data), \
            'Data amount mismatch: {} LAED points vs {} dialog turns'.format(self.laed_z_dialog.shape[0],
                                                                             len(self.data))
        self.laed_z_seed = laed_z_seed

        # prepare indexes for warm up
        self.warmup_data = warmup_data
        if self.warmup_data is not None:
            self.warmup_size = len(self.warmup_data)
            self.warmup_indexes = range(self.warmup_size)
            assert self.laed_z_seed.shape[0] == len(self.warmup_data), \
                'Data amount mismatch: {} LAED seed points vs {} warmup turns'.format(self.laed_z_seed.shape[0],
                                                                                      len(self.warmup_data))
        self.warmup_flags = None
        self.warmup_num_batch = None

    def flatten_dialog(self, data, laed_z, backward_size):
        results = []
        laed_z_flat = []
        for dialog, dialog_laed_z in zip(data, laed_z):
            for i in range(1, len(dialog)):
                e_id = i
                s_id = max(0, e_id - backward_size)
                response = dialog[i].copy()
                if response.speaker == USR:
                    continue
                response['utt'] = self.pad_to(self.max_utt_size, response.utt, do_pad=False)

                contexts = []
                for turn in dialog[s_id:e_id]:
                    turn['utt'] = self.pad_to(self.max_utt_size, turn.utt, do_pad=False)
                    contexts.append(turn)
                results.append(Pack(context=contexts, response=response))
                laed_z_flat.append(dialog_laed_z[i])
        return results, np.array(laed_z_flat)

    def epoch_init(self, config, shuffle=True, verbose=True):
        super(ZslMaluubaDataLoader, self).epoch_init(config, shuffle, verbose)
        self.warmup_flags = [False] * self.num_batch

        if self.warmup_data is None:
            return

        self.warmup_num_batch = int(self.warmup_size / config.batch_size)
        for i in range(self.warmup_num_batch):
            self.batch_indexes.append(np.random.choice(self.warmup_indexes, config.batch_size, replace=False))
            self.warmup_flags.append(True)

        if shuffle:
            temp_batch_id = range(len(self.warmup_flags))
            np.random.shuffle(temp_batch_id)
            self.batch_indexes = [self.batch_indexes[i] for i in temp_batch_id]
            self.warmup_flags = [self.warmup_flags[i] for i in temp_batch_id]

        if verbose:
            self.logger.info("%s add with %d warm up batches" % (self.name, self.warmup_num_batch))

    def next_batch(self):
        if self.ptr < self.num_batch:
            is_warmup = self.warmup_flags[self.ptr]
            selected_ids = self.batch_indexes[self.ptr]
            self.ptr += 1

            if is_warmup:
                return self._prepare_warmup_batch(selected_ids)
            else:
                return self._prepare_batch(selected_ids)
        else:
            return None

    def _prepare_batch(self, selected_index):
        # the batch index, the starting point and end point for segment
        rows = [self.data[idx] for idx in selected_index]
        laed_z = self.laed_z_dialog[selected_index, :]

        cxt_lens, ctx_utts = [], []
        out_utts, out_lens = [], []
        domains, domain_metas = [], []

        for row in rows:
            in_row, out_row = row.context, row.response

            # source context
            batch_ctx = []
            #for item in out_row.kb:
            #    batch_ctx.append(item)
            for turn in in_row:
                batch_ctx.append(self.pad_to(self.max_utt_size, turn.utt))

            cxt_lens.append(len(batch_ctx))
            ctx_utts.append(batch_ctx)

            # target response
            out_utt = [t for idx, t in enumerate(out_row.utt)]
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))
            domains.append(out_row.domain)
            domain_metas.append(out_row.domain_id)

        domain_metas = np.array(domain_metas)
        vec_ctx_lens = np.array(cxt_lens)
        max_ctx_len = np.max(vec_ctx_lens)
        vec_ctx_utts = np.zeros((self.batch_size, max_ctx_len, self.max_utt_size), dtype=np.int32)
        vec_ctx_confs = np.ones((self.batch_size, max_ctx_len), dtype=np.float32)

        vec_out_utts = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_out_lens = np.array(out_lens)

        for b_id in range(self.batch_size):
            vec_out_utts[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            vec_ctx_utts[b_id, 0:vec_ctx_lens[b_id], :] = ctx_utts[b_id]

        return Pack(context_lens=vec_ctx_lens,
                    contexts=vec_ctx_utts,
                    context_confs=vec_ctx_confs,
                    output_lens=vec_out_lens,
                    outputs=vec_out_utts,
                    domains=domains,
                    domain_metas=domain_metas,
                    laed_z=laed_z)

    def _prepare_warmup_batch(self, selected_ids):
        # the batch index, the starting point and end point for segment
        rows = [self.warmup_data[idx] for idx in selected_ids]
        out_utts, out_lens = [], []
        out_acts, out_act_lens = [], []
        domains, domain_metas = [], []

        laed_z = self.laed_z_seed[selected_ids, :]
        for row in rows:
            out_utt = [t for idx, t in enumerate(row.utt)]

            # target response
            out_acts.append(row.actions)
            out_act_lens.append(len(row.actions))

            out_utts.append(out_utt)
            out_lens.append(len(out_utt))

            domains.append(row.domain)
            domain_metas.append(row.domain_id)

        vec_out_lens = np.array(out_lens)
        domain_metas = np.array(domain_metas)
        vec_out_utts = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_out_acts = np.zeros((self.batch_size, np.max(out_act_lens)), dtype=np.int32)

        for b_id in range(self.batch_size):
            vec_out_utts[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            vec_out_acts[b_id, 0:out_act_lens[b_id]] = out_acts[b_id]

        return Pack(output_lens=vec_out_lens,
                    outputs=vec_out_utts,
                    output_actions=vec_out_acts,
                    domains=domains,
                    domain_metas=domain_metas,
                    laed_z=laed_z)


class ZslLASMDDialDataLoader(DataLoader):
    def __init__(self, name, data, config, warmup_data=None):
        super(ZslLASMDDialDataLoader, self).__init__(name)
        self.max_utt_size = config.max_utt_len

        self.data = self.flatten_dialog(data, config.backward_size)
        self.data_size = len(self.data)
        data_lens = [len(line.context) for line in self.data]
        if False:
            self.indexes = list(np.argsort(data_lens))[::-1]
        else:
            self.indexes = range(len(data_lens))

        # prepare indexes for warm up
        self.warmup_data = warmup_data
        if self.warmup_data is not None:
            self.warmup_size = len(self.warmup_data)
            self.warmup_indexes = range(self.warmup_size)
        self.warmup_flags = None
        self.warmup_num_batch = None

    def flatten_dialog(self, data, backward_size):
        results = []
        for dialog in data:
            for i in range(1, len(dialog)):
                e_id = i
                s_id = max(0, e_id - backward_size)
                response = dialog[i].copy()
                if response.speaker == USR:
                    continue
                response['utt'] = self.pad_to(self.max_utt_size, response.utt, do_pad=False)
                response['kb'] = [self.pad_to(self.max_utt_size, item, do_pad=True) for item in response.kb]
                response['laed_z'] = np.array(response.laed_z)

                contexts = []
                for turn in dialog[s_id:e_id]:
                    turn['utt'] = self.pad_to(self.max_utt_size, turn.utt, do_pad=False)
                    contexts.append(turn)
                results.append(Pack(context=contexts, response=response))
        return results

    def epoch_init(self, config, shuffle=True, verbose=True):
        super(ZslLASMDDialDataLoader, self).epoch_init(config, shuffle, verbose)
        self.warmup_flags = [False] * self.num_batch

        if self.warmup_data is None:
            return

        self.warmup_num_batch = int(self.warmup_size / config.batch_size)
        for i in range(self.warmup_num_batch):
            self.batch_indexes.append(np.random.choice(self.warmup_indexes, config.batch_size, replace=False))
            self.warmup_flags.append(True)

        if shuffle:
            temp_batch_id = range(len(self.warmup_flags))
            np.random.shuffle(temp_batch_id)
            self.batch_indexes = [self.batch_indexes[i] for i in temp_batch_id]
            self.warmup_flags = [self.warmup_flags[i] for i in temp_batch_id]

        if verbose:
            self.logger.info("%s add with %d warm up batches" % (self.name, self.warmup_num_batch))

    def next_batch(self):
        if self.ptr < self.num_batch:
            is_warmup = self.warmup_flags[self.ptr]
            selected_ids = self.batch_indexes[self.ptr]
            self.ptr += 1

            if is_warmup:
                return self._prepare_warmup_batch(selected_ids)
            else:
                return self._prepare_batch(selected_ids)
        else:
            return None

    def _prepare_batch(self, selected_index):
        # the batch index, the starting point and end point for segment
        rows = [self.data[idx] for idx in selected_index]

        cxt_lens, ctx_utts = [], []
        out_utts, out_lens = [], []
        domains, domain_metas = [], []

        laed_z = []
        for row in rows:
            in_row, out_row = row.context, row.response

            # source context
            batch_ctx = []
            for item in out_row.kb:
                batch_ctx.append(item)
            for turn in in_row:
                batch_ctx.append(self.pad_to(self.max_utt_size, turn.utt))

            cxt_lens.append(len(batch_ctx))
            ctx_utts.append(batch_ctx)

            # target response
            out_utt = [t for idx, t in enumerate(out_row.utt)]
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))
            domains.append(out_row.domain)
            domain_metas.append(out_row.domain_id)

            laed_z.append(row.response.laed_z)
        laed_z = np.array(laed_z)

        domain_metas = np.array(domain_metas)
        vec_ctx_lens = np.array(cxt_lens)
        max_ctx_len = np.max(vec_ctx_lens)
        vec_ctx_utts = np.zeros((self.batch_size, max_ctx_len, self.max_utt_size), dtype=np.int32)
        vec_ctx_confs = np.ones((self.batch_size, max_ctx_len), dtype=np.float32)

        vec_out_utts = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_out_lens = np.array(out_lens)

        for b_id in range(self.batch_size):
            vec_out_utts[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            vec_ctx_utts[b_id, 0:vec_ctx_lens[b_id], :] = ctx_utts[b_id]

        return Pack(context_lens=vec_ctx_lens,
                    contexts=vec_ctx_utts,
                    context_confs=vec_ctx_confs,
                    output_lens=vec_out_lens,
                    outputs=vec_out_utts,
                    domains=domains,
                    domain_metas=domain_metas,
                    laed_z=laed_z)

    def _prepare_warmup_batch(self, selected_ids):
        # the batch index, the starting point and end point for segment
        rows = [self.warmup_data[idx] for idx in selected_ids]
        out_utts, out_lens = [], []
        out_acts, out_act_lens = [], []
        domains, domain_metas = [], []
        laed_z = []
        for row in rows:
            out_utt = [t for idx, t in enumerate(row.utt)]

            # target response
            out_acts.append(row.actions)
            out_act_lens.append(len(row.actions))

            out_utts.append(out_utt)
            out_lens.append(len(out_utt))

            domains.append(row.domain)
            domain_metas.append(row.domain_id)

            laed_z.append(row.laed_z)
        laed_z = np.array(laed_z)

        vec_out_lens = np.array(out_lens)
        domain_metas = np.array(domain_metas)
        vec_out_utts = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_out_acts = np.zeros((self.batch_size, np.max(out_act_lens)), dtype=np.int32)

        for b_id in range(self.batch_size):
            vec_out_utts[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            vec_out_acts[b_id, 0:out_act_lens[b_id]] = out_acts[b_id]

        return Pack(output_lens=vec_out_lens,
                    outputs=vec_out_utts,
                    output_actions=vec_out_acts,
                    domains=domains,
                    domain_metas=domain_metas,
                    laed_z=laed_z)
