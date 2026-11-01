"""Compact sequence and ordered binary-tree encoder/attention-decoder models."""

from __future__ import annotations

from math import sqrt
import torch
from torch import nn
from torch.nn import functional as F

from tsfmxai.symbolic_regression.canonical_data import PAD, BOS, EOS, END_TERM


def collate_canonical(examples):
    batch = {}
    node_length = max(len(e['tree_ids']) for e in examples)
    target_length = max(len(e['target']) for e in examples)
    for key in ('prefix_ids', 'tree_ids', 'prefix_values', 'tree_values',
                'left_children', 'right_children', 'target', 'coefficient_targets'):
        is_target = key in ('target', 'coefficient_targets')
        length = target_length if is_target else node_length
        dtype = torch.float32 if 'values' in key or key == 'coefficient_targets' else torch.long
        fill = -1 if 'children' in key else 0
        tensor = torch.full((len(examples), length), fill, dtype=dtype)
        for i, example in enumerate(examples):
            tensor[i, :len(example[key])] = torch.tensor(example[key], dtype=dtype)
        batch[key] = tensor
    batch['lengths'] = torch.tensor([len(e['tree_ids']) for e in examples])
    return batch


class CanonicalEncoderDecoder(nn.Module):
    def __init__(self, vocab_size, encoder='transformer', predict_coefficients=False,
                 dimension=96, heads=4, encoder_layers=2, dropout=0.1, max_nodes=13):
        super().__init__()
        if encoder not in ('transformer', 'tree_lstm'):
            raise ValueError('Unknown encoder architecture.')
        self.encoder_kind = encoder
        self.predict_coefficients = predict_coefficients
        self.dimension = dimension
        self.source_embedding = nn.Embedding(vocab_size, dimension, padding_idx=PAD)
        self.numeric_embedding = nn.Linear(1, dimension, bias=False)
        if encoder == 'transformer':
            self.position = nn.Embedding(max_nodes, dimension)
            layer = nn.TransformerEncoderLayer(dimension, heads, dimension * 3, dropout,
                                               batch_first=True, norm_first=True)
            self.sequence_encoder = nn.TransformerEncoder(layer, encoder_layers,
                                                           enable_nested_tensor=False)
        else:
            # Ordered N-ary TreeLSTM specialized to two children: i, o, u, f_left, f_right.
            self.tree_gates = nn.Linear(3 * dimension, 5 * dimension)
        self.source_norm = nn.LayerNorm(dimension)
        self.target_embedding = nn.Embedding(vocab_size, dimension, padding_idx=PAD)
        self.decoder = nn.GRU(dimension, dimension, batch_first=True)
        self.initial = nn.Linear(dimension, dimension)
        self.query = nn.Linear(dimension, dimension, bias=False)
        self.combine = nn.Linear(2 * dimension, dimension)
        self.dropout = nn.Dropout(dropout)
        self.tokens = nn.Linear(dimension, vocab_size)
        self.coefficients = nn.Linear(dimension, 1) if predict_coefficients else None

    def encode(self, batch):
        if self.encoder_kind == 'transformer':
            ids, values = batch['prefix_ids'], batch['prefix_values']
            mask = ids.eq(PAD)
            x = self.source_embedding(ids) + self.numeric_embedding(torch.asinh(values).unsqueeze(-1))
            x = x + self.position(torch.arange(ids.shape[1], device=ids.device))[None]
            memory = self.source_norm(self.sequence_encoder(x, src_key_padding_mask=mask))
            root = (memory * (~mask).unsqueeze(-1)).sum(1) / batch['lengths'].unsqueeze(-1)
        else:
            ids, values = batch['tree_ids'], batch['tree_values']
            mask = ids.eq(PAD)
            x = self.source_embedding(ids) + self.numeric_embedding(torch.asinh(values).unsqueeze(-1))
            zero = x.new_zeros((x.shape[0], self.dimension))
            states, cells = [zero], [zero]
            rows = torch.arange(x.shape[0], device=x.device)
            for position in range(x.shape[1]):
                hstack, cstack = torch.stack(states, dim=1), torch.stack(cells, dim=1)
                left = batch['left_children'][:, position] + 1
                right = batch['right_children'][:, position] + 1
                hl, hr = hstack[rows, left], hstack[rows, right]
                cl, cr = cstack[rows, left], cstack[rows, right]
                gates = self.tree_gates(torch.cat((x[:, position], hl, hr), dim=-1))
                i, o, u, fl, fr = gates.chunk(5, dim=-1)
                c = torch.sigmoid(i) * torch.tanh(u) + torch.sigmoid(fl) * cl + torch.sigmoid(fr) * cr
                valid = (~mask[:, position]).unsqueeze(-1)
                h = torch.sigmoid(o) * torch.tanh(c) * valid
                states.append(h)
                cells.append(c * valid)
            memory = self.source_norm(torch.stack(states[1:], dim=1))
            root = memory[rows, batch['lengths'] - 1]
        return memory, mask, torch.tanh(self.initial(root)).unsqueeze(0)

    def _heads(self, states, memory, mask):
        weights = torch.matmul(self.query(states), memory.transpose(1, 2)) / sqrt(self.dimension)
        weights = weights.masked_fill(mask[:, None, :], float('-inf')).softmax(-1)
        context = torch.matmul(weights, memory)
        features = self.dropout(torch.tanh(self.combine(torch.cat((states, context), dim=-1))))
        logits = self.tokens(features)
        coefficients = self.coefficients(features).squeeze(-1) if self.coefficients else None
        return logits, coefficients

    def forward(self, batch, decoder_tokens=None):
        memory, mask, hidden = self.encode(batch)
        if decoder_tokens is None:
            decoder_tokens = batch['target'][:, :-1]
        states, _ = self.decoder(self.target_embedding(decoder_tokens), hidden)
        return self._heads(states, memory, mask)

    @torch.no_grad()
    def generate(self, batch, max_steps=64):
        """Greedy generation uses source features only; no target prefixes."""
        memory, mask, hidden = self.encode(batch)
        token = torch.full((memory.shape[0], 1), BOS, dtype=torch.long, device=memory.device)
        finished = torch.zeros(memory.shape[0], dtype=torch.bool, device=memory.device)
        outputs, values = [], []
        for _ in range(max_steps):
            state, hidden = self.decoder(self.target_embedding(token), hidden)
            logits, coefficient = self._heads(state, memory, mask)
            logits[:, :, PAD] = float('-inf')
            logits[:, :, BOS] = float('-inf')
            token = logits.argmax(-1)
            token = torch.where(finished[:, None], torch.full_like(token, EOS), token)
            outputs.append(token[:, 0])
            if coefficient is not None:
                values.append(torch.sinh(coefficient[:, 0]))
            finished |= token[:, 0].eq(EOS)
            if bool(finished.all()):
                break
        return torch.stack(outputs, dim=1), torch.stack(values, dim=1) if values else None


def supervised_loss(model, batch, coefficient_weight=1.0):
    logits, coefficients = model(batch)
    labels = batch['target'][:, 1:]
    token_sum = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                               ignore_index=PAD, reduction='sum')
    token_count = labels.ne(PAD).sum()
    term_mask = labels.eq(END_TERM)
    term_count = term_mask.sum()
    coefficient_sum = token_sum.new_zeros(())
    coefficient_mae = token_sum.new_zeros(())
    if coefficients is not None and bool(term_count):
        truth = batch['coefficient_targets'][:, 1:][term_mask]
        predicted = coefficients[term_mask]
        coefficient_sum = F.smooth_l1_loss(predicted, torch.asinh(truth), reduction='sum')
        coefficient_mae = (torch.sinh(predicted) - truth).abs().sum()
    loss = token_sum / token_count.clamp_min(1)
    loss = loss + coefficient_weight * coefficient_sum / term_count.clamp_min(1)
    metrics = {'token_sum': float(token_sum.detach()), 'token_count': int(token_count),
               'coefficient_sum': float(coefficient_sum.detach()), 'term_count': int(term_count),
               'coefficient_mae_sum': float(coefficient_mae.detach()),
               'token_correct': int((logits.argmax(-1).eq(labels) & labels.ne(PAD)).sum())}
    return loss, metrics
