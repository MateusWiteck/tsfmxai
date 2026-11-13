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


# Parameter groups that belong to the encoder; the names are the same in the full model
# and in CanonicalEncoder, so one state dict slice loads into either.
ENCODER_PREFIXES = ('source_embedding', 'numeric_embedding', 'position', 'sequence_encoder',
                    'tree_gates', 'source_norm')


def encoder_state(state_dict):
    """Return the encoder entries of a CanonicalEncoderDecoder state dict."""
    return {key: value for key, value in state_dict.items()
            if key.split('.', 1)[0] in ENCODER_PREFIXES}


def _build_encoder(module, vocab_size, encoder, dimension, heads, encoder_layers, dropout, max_nodes):
    if encoder not in ('transformer', 'tree_lstm'):
        raise ValueError('Unknown encoder architecture.')
    module.encoder_kind = encoder
    module.dimension = dimension
    module.max_nodes = max_nodes
    module.source_embedding = nn.Embedding(vocab_size, dimension, padding_idx=PAD)
    module.numeric_embedding = nn.Linear(1, dimension, bias=False)
    if encoder == 'transformer':
        # max_nodes is the size of this table: the longest input tree the encoder accepts.
        module.position = nn.Embedding(max_nodes, dimension)
        layer = nn.TransformerEncoderLayer(dimension, heads, dimension * 3, dropout,
                                           batch_first=True, norm_first=True)
        module.sequence_encoder = nn.TransformerEncoder(layer, encoder_layers,
                                                         enable_nested_tensor=False)
    else:
        # Ordered N-ary TreeLSTM specialized to two children: i, o, u, f_left, f_right.
        module.tree_gates = nn.Linear(3 * dimension, 5 * dimension)
    module.source_norm = nn.LayerNorm(dimension)


def _encode_features(module, batch):
    """Per-node memory, padding mask and the root summary (before the decoder bridge)."""
    if module.encoder_kind == 'transformer':
        ids, values = batch['prefix_ids'], batch['prefix_values']
        mask = ids.eq(PAD)
        x = module.source_embedding(ids) + module.numeric_embedding(torch.asinh(values).unsqueeze(-1))
        x = x + module.position(torch.arange(ids.shape[1], device=ids.device))[None]
        memory = module.source_norm(module.sequence_encoder(x, src_key_padding_mask=mask))
        root = (memory * (~mask).unsqueeze(-1)).sum(1) / batch['lengths'].unsqueeze(-1)
    else:
        ids, values = batch['tree_ids'], batch['tree_values']
        mask = ids.eq(PAD)
        x = module.source_embedding(ids) + module.numeric_embedding(torch.asinh(values).unsqueeze(-1))
        zero = x.new_zeros((x.shape[0], module.dimension))
        states, cells = [zero], [zero]
        rows = torch.arange(x.shape[0], device=x.device)
        for position in range(x.shape[1]):
            hstack, cstack = torch.stack(states, dim=1), torch.stack(cells, dim=1)
            left = batch['left_children'][:, position] + 1
            right = batch['right_children'][:, position] + 1
            hl, hr = hstack[rows, left], hstack[rows, right]
            cl, cr = cstack[rows, left], cstack[rows, right]
            gates = module.tree_gates(torch.cat((x[:, position], hl, hr), dim=-1))
            i, o, u, fl, fr = gates.chunk(5, dim=-1)
            c = torch.sigmoid(i) * torch.tanh(u) + torch.sigmoid(fl) * cl + torch.sigmoid(fr) * cr
            valid = (~mask[:, position]).unsqueeze(-1)
            h = torch.sigmoid(o) * torch.tanh(c) * valid
            states.append(h)
            cells.append(c * valid)
        memory = module.source_norm(torch.stack(states[1:], dim=1))
        root = memory[rows, batch['lengths'] - 1]
    return memory, mask, root


class CanonicalEncoder(nn.Module):
    """The encoder of CanonicalEncoderDecoder alone: AST batch -> (memory, mask, root).

    ``memory`` has shape (batch, nodes, dimension), ``mask`` marks padding, and ``root`` is
    the (batch, dimension) summary of the whole tree.
    """

    def __init__(self, vocab_size, encoder='transformer', dimension=96, heads=4,
                 encoder_layers=2, dropout=0.1, max_nodes=13, **_ignored):
        super().__init__()
        _build_encoder(self, vocab_size, encoder, dimension, heads, encoder_layers, dropout, max_nodes)

    def forward(self, batch):
        return _encode_features(self, batch)

    @classmethod
    def from_pretrained(cls, directory, *, revision=None):
        """Load a local release folder or a Hugging Face model repository ID.

        Repository releases must contain ``config.json`` and
        ``<variant>/config.json`` plus ``<variant>/encoder.safetensors`` (or
        the corresponding files under ``models/<variant>/``). A flat release
        with files at its root is also supported. Pass ``revision`` to pin a
        Hugging Face commit; local paths ignore it.
        """
        import json
        from pathlib import Path
        from safetensors.torch import load_file

        variant = None
        if isinstance(directory, tuple):
            if len(directory) != 2:
                raise ValueError("directory tuple must be (repo_or_path, variant)")
            directory, variant = directory
        source = directory
        local_directory = Path(directory) if isinstance(directory, (str, Path)) else None
        if local_directory is not None and local_directory.exists():
            candidates = ([local_directory / variant] if variant else [])
            candidates += ([local_directory / 'models' / variant] if variant else [])
            candidates += [local_directory]
            release_dir = next(
                (candidate for candidate in candidates
                 if (candidate / 'config.json').exists()
                 and (candidate / 'encoder.safetensors').exists()),
                None,
            )
            if release_dir is None:
                raise FileNotFoundError(
                    f'No config.json and encoder.safetensors found under {local_directory}'
                )
            config_path = release_dir / 'config.json'
            weights_path = release_dir / 'encoder.safetensors'
        else:
            from huggingface_hub import hf_hub_download

            repo_id = str(source)
            prefixes = ([variant, f'models/{variant}'] if variant else []) + ['']
            config_path = weights_path = None
            last_error = None
            for prefix in prefixes:
                config_name = f'{prefix}/config.json' if prefix else 'config.json'
                weights_name = (f'{prefix}/encoder.safetensors'
                                if prefix else 'encoder.safetensors')
                try:
                    config_path = Path(hf_hub_download(
                        repo_id, config_name, revision=revision
                    ))
                    weights_path = Path(hf_hub_download(
                        repo_id, weights_name, revision=revision
                    ))
                    break
                except Exception as error:
                    config_path = weights_path = None
                    last_error = error
            if weights_path is None:
                detail = f' at revision {revision}' if revision else ''
                raise FileNotFoundError(
                    f'No encoder release found in {repo_id}{detail}; '
                    'pass a valid variant if required.'
                ) from last_error
        if not config_path.exists() or not weights_path.exists():
            raise FileNotFoundError('Encoder release requires config.json and encoder.safetensors')
        config = json.loads(config_path.read_text(encoding='utf-8'))
        model = cls(**config['model_kwargs'])
        model.load_state_dict(load_file(str(weights_path), device='cpu'))
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model.eval()


class CanonicalEncoderDecoder(nn.Module):
    def __init__(self, vocab_size, encoder='transformer', predict_coefficients=False,
                 dimension=96, heads=4, encoder_layers=2, dropout=0.1, max_nodes=13):
        super().__init__()
        _build_encoder(self, vocab_size, encoder, dimension, heads, encoder_layers, dropout, max_nodes)
        self.predict_coefficients = predict_coefficients
        self.target_embedding = nn.Embedding(vocab_size, dimension, padding_idx=PAD)
        self.decoder = nn.GRU(dimension, dimension, batch_first=True)
        self.initial = nn.Linear(dimension, dimension)
        self.query = nn.Linear(dimension, dimension, bias=False)
        self.combine = nn.Linear(2 * dimension, dimension)
        self.dropout = nn.Dropout(dropout)
        self.tokens = nn.Linear(dimension, vocab_size)
        self.coefficients = nn.Linear(dimension, 1) if predict_coefficients else None

    def encode(self, batch):
        memory, mask, root = _encode_features(self, batch)
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
