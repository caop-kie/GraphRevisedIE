from typing import *
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from . import resnet


class Encoder(nn.Module):

    def __init__(self,
                 char_embedding_dim: int,
                 out_dim: int,
                 image_feature_dim: int = 512,
                 nheaders: int = 8,
                 nlayers: int = 6,
                 feedforward_dim: int = 2048,
                 dropout: float = 0.1,
                 max_len: int = 100,
                 roi_pooling_size: Tuple[int, int] = (7, 7)):
        super().__init__()
        self.dropout = dropout
        self.roi_pooling_size = tuple(roi_pooling_size)  # (h, w)

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=char_embedding_dim,
                                                               nhead=nheaders,
                                                               dim_feedforward=feedforward_dim,
                                                               dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=nlayers)
        self.cnn = resnet.resnet50(output_channels=image_feature_dim)

        self.conv = nn.Conv2d(image_feature_dim, out_dim, self.roi_pooling_size)
        self.bn = nn.BatchNorm2d(out_dim)

        self.projection = nn.Linear(2 * out_dim, out_dim)  # 2 * 512, 512
        self.norm = nn.LayerNorm(out_dim)

        # Compute the positional encodings once in log space.
        position_embedding = torch.zeros(max_len, char_embedding_dim)    # 100,  512
        position = torch.arange(0, max_len).unsqueeze(1).float()  #[0, 1, 2, ..., 99] -> [[0], [1], [2], ..., [99]]
        div_term = torch.exp(torch.arange(0, char_embedding_dim, 2).float() *
                             -(math.log(10000.0) / char_embedding_dim))
        position_embedding[:, 0::2] = torch.sin(position * div_term)
        position_embedding[:, 1::2] = torch.cos(position * div_term)
        position_embedding = position_embedding.unsqueeze(0).unsqueeze(0)  # 1, 1, max_len, char_embedding_dim
        self.register_buffer('position_embedding', position_embedding)
        self.pe_dropout = nn.Dropout(self.dropout)    # 0.1
        self.output_dropout = nn.Dropout(self.dropout)

        self.dim_bbox_sinusoid_emb = 1024
        self.dim_bbox_projection = 512
        self.bbox_sinusoid_emb = PositionalEmbedding2D(self.dim_bbox_sinusoid_emb, dim_bbox=8)
        self.bbox_projection = nn.Linear(
            self.dim_bbox_sinusoid_emb, self.dim_bbox_projection, bias=False
        )

    def forward(self, images, boxes_coordinate, texts_emb, key_padding_mask):
        B, N, T, D = texts_emb.shape
        _, _, origin_H, origin_W = images.shape

        images = self.cnn(images)
        _, C, H, W = images.shape

        rois_batch = torch.zeros(B, N, 5, device=images.device)
        for i in range(B):  # (B, N, 8)
            # (N, 8)
            doc_boxes = boxes_coordinate[i]  #[10,20,30,40,50,60,70,80]
            # (N, 4)
            pos = torch.stack([doc_boxes[:, 0], doc_boxes[:, 1], doc_boxes[:, 4], doc_boxes[:, 5]], dim=1) #[10,20,50,60]
            rois_batch[i, :, 1:5] = pos
            rois_batch[i, :, 0] = i

        spatial_scale = float(H / origin_H)
        image_emb = roi_align(images, rois_batch.view(-1, 5), self.roi_pooling_size, spatial_scale)
        image_emb = F.relu(self.bn(self.conv(image_emb))).squeeze().unsqueeze(dim=1)

        normalized_bbox = torch.zeros(B, N, 8, device=boxes_coordinate.device)
        normalized_bbox[..., 0::2] = boxes_coordinate[..., 0::2]/origin_W
        normalized_bbox[..., 1::2] = boxes_coordinate[..., 1::2]/origin_H
        normalized_bbox = normalized_bbox * 100  # BROS scaling ratio
        bbox_pos_embedding = self.calc_bbox_pos_emb(normalized_bbox)

        rel_pos_embedding = bbox_pos_embedding[:, 0].transpose(0, 1).unsqueeze(dim=2).expand(B, N, T, D)
        rel_pos_embedding = rel_pos_embedding.masked_fill(key_padding_mask.view(B, N, T, 1), 0)
        rel_pos_embedding = rel_pos_embedding.reshape(B*N, T, D)

        embeddings = texts_emb + self.position_embedding[:, :, :texts_emb.size(2), :]
        segment_embeddings = self.pe_dropout(embeddings).reshape(B * N, T, D)
        multimodal_emb = (image_emb.expand_as(segment_embeddings) + segment_embeddings + rel_pos_embedding).transpose(0, 1).contiguous()

        multimodal_emb = self.transformer_encoder(multimodal_emb, src_key_padding_mask=key_padding_mask).transpose(0, 1).contiguous()
        multimodal_emb = self.output_dropout(self.norm(multimodal_emb))
        return multimodal_emb

    # B, N, 8
    def calc_bbox_pos_emb(self, bbox):
        # bbox_t: [seq_length, batch_size, dim_bbox]
        # bbox_t: [seg_length, batch_size, dim_bbox]
        bbox_t = bbox.transpose(0, 1)

        bbox_pos = bbox_t[None, :, :, :] - bbox_t[:, None, :, :]

        bbox_pos_emb = self.bbox_sinusoid_emb(bbox_pos)
        bbox_pos_emb = self.bbox_projection(bbox_pos_emb)

        # N N B 512
        return bbox_pos_emb


class PositionalEmbedding2D(nn.Module):
    def __init__(self, demb, dim_bbox=8):
        super(PositionalEmbedding2D, self).__init__()

        self.demb = demb
        self.dim_bbox = dim_bbox

        self.x_pos_emb = PositionalEmbedding1D(demb // dim_bbox)
        self.y_pos_emb = PositionalEmbedding1D(demb // dim_bbox)

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, bbox):
        # bbox: [seg_length, batch_size, dim_bbox]
        stack = []
        for i in range(self.dim_bbox):
            if i % 2 == 0:
                stack.append(self.x_pos_emb(bbox[..., i]))
            else:
                stack.append(self.y_pos_emb(bbox[..., i]))
        bbox_pos_emb = torch.cat(stack, dim=-1)
        return bbox_pos_emb


class PositionalEmbedding1D(nn.Module):
    # Reference: https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py#L15

    def __init__(self, demb):
        super(PositionalEmbedding1D, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        seq_size = pos_seq.size()

        if len(seq_size) == 2:
            b1, b2 = seq_size
            sinusoid_inp = pos_seq.view(b1, b2, 1) * self.inv_freq.view(
                1, 1, self.demb // 2
            )
        elif len(seq_size) == 3:
            b1, b2, b3 = seq_size
            sinusoid_inp = pos_seq.view(b1, b2, b3, 1) * self.inv_freq.view(
                1, 1, 1, self.demb // 2
            )
        else:
            raise ValueError(f"Invalid seq_size={len(seq_size)}")

        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        return pos_emb