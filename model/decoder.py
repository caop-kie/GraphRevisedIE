from typing import *

import torch
import torch.nn as nn
from torch import Tensor

from .crf import ConditionalRandomField
from utils.class_utils import keys_vocab_cls, iob_labels_vocab_cls


class MLPLayer(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: Optional[int] = None,
                 hidden_dims: Optional[List[int]] = None,
                 layer_norm: bool = False,
                 dropout: Optional[float] = 0.0,
                 activation: Optional[str] = 'relu'):
        super().__init__()
        layers = []
        activation_layer = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU
        }

        if hidden_dims:
            for dim in hidden_dims:
                layers.append(nn.Linear(in_dim, dim))
                layers.append(activation_layer.get(activation, nn.Identity()))
                if layer_norm:
                    layers.append(nn.LayerNorm(dim))
                if dropout:
                    layers.append(nn.Dropout(dropout))
                in_dim = dim

        if not out_dim:
            layers.append(nn.Identity())
        else:
            layers.append(nn.Linear(in_dim, out_dim))

        self.mlp = nn.Sequential(*layers)
        self.out_dim = out_dim if out_dim else hidden_dims[-1]

    def forward(self, *input: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat(input, 1))


class Decoder(nn.Module):

    def __init__(self, bilstm_kwargs, mlp_kwargs, crf_kwargs):
        super().__init__()
        self.lstm = nn.LSTM(**bilstm_kwargs)
        self.mlp = MLPLayer(**mlp_kwargs)
        self.crf_layer = ConditionalRandomField(**crf_kwargs)

    def sort_tensor(self, x: torch.Tensor, length: torch.Tensor):
        sorted_lenght, sorted_order = torch.sort(length, descending=True)
        _, invert_order = sorted_order.sort(0, descending=False)
        return x[sorted_order], sorted_lenght, invert_order

    def forward(self, emb, segment_emb, mask, length, tags):
        B, N, T, D = emb.shape
        emb = emb.reshape(B, N * T, -1)
        mask = mask.reshape(B, N * T)

        segment_emb = segment_emb.unsqueeze(2).expand(B, N, T, -1).reshape(B, N * T, -1)
        emb = segment_emb + emb
        doc_seq_len = length.sum(dim=-1)

        max_doc_seq_len = doc_seq_len.max()
        new_emb = torch.zeros_like(emb, device=emb.device)
        new_mask = torch.zeros_like(mask, device=emb.device)
        if self.training:
            tags = tags.reshape(B, N * T)
            new_tag = torch.full_like(tags, iob_labels_vocab_cls.stoi['<pad>'], device=emb.device)
            new_tag = new_tag[:, :max_doc_seq_len]

        for i in range(B):
            doc_x = emb[i]
            doc_mask = mask[i]
            valid_doc_x = doc_x[doc_mask == 1]
            num_valid = valid_doc_x.size(0)
            new_emb[i, :num_valid] = valid_doc_x
            new_mask[i, :doc_seq_len[i]] = 1

            if self.training:
                valid_tag = tags[i][doc_mask == 1]
                new_tag[i, :num_valid] = valid_tag

        new_emb = new_emb[:, :max_doc_seq_len, :]
        new_mask = new_mask[:, :max_doc_seq_len]

        if not self.training:
            new_tag = None

        seq, sorted_lengths, invert_order = self.sort_tensor(new_emb, doc_seq_len)
        packed_seq = nn.utils.rnn.pack_padded_sequence(seq, lengths=sorted_lengths.cpu(), batch_first=True)
        self.lstm.flatten_parameters()
        output, _ = self.lstm(packed_seq)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True,
                                                     padding_value=keys_vocab_cls.stoi['<pad>'])
        output = output[invert_order]
        logits = self.mlp(output)

        log_likelihood = None
        if self.training:
            # (B,)
            log_likelihood = self.crf_layer(logits,
                                            new_tag,
                                            mask=new_mask,
                                            input_batch_first=True,
                                            keepdim=True)

        return logits, new_mask, log_likelihood