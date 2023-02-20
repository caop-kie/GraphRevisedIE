import torch
import torch.nn as nn
import numpy as np

from .encoder import Encoder
from .grcn import GRCN
from .decoder import Decoder
from utils.class_utils import keys_vocab_cls, iob_labels_vocab_cls


class Model(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        embedding_kwargs = kwargs['embedding_kwargs']
        encoder_kwargs = kwargs['encoder_kwargs']
        decoder_kwargs = kwargs['decoder_kwargs']
        self.make_model(embedding_kwargs, encoder_kwargs, decoder_kwargs)

    def make_model(self, embedding_kwargs, encoder_kwargs, decoder_kwargs):
        embedding_kwargs['num_embeddings'] = len(keys_vocab_cls)    # 6382
        self.character_emb = nn.Embedding(**embedding_kwargs)

        encoder_kwargs['char_embedding_dim'] = embedding_kwargs['embedding_dim']
        self.encoder = Encoder(**encoder_kwargs)

        D = encoder_kwargs['out_dim']
        self.graph = GRCN(D, D, 'cuda', 1024, D, D, 'dense')

        decoder_kwargs['bilstm_kwargs']['input_size'] = encoder_kwargs['out_dim']
        decoder_kwargs['mlp_kwargs']['in_dim'] = decoder_kwargs['bilstm_kwargs']['hidden_size'] * 2
        decoder_kwargs['mlp_kwargs']['out_dim'] = len(iob_labels_vocab_cls)
        decoder_kwargs['crf_kwargs']['num_tags'] = len(iob_labels_vocab_cls)
        self.decoder = Decoder(**decoder_kwargs)

    def aggregate(self, input, mask):
        input = input * mask.detach().unsqueeze(2).float()
        sum_out = torch.sum(input, dim=1)
        text_len = mask.float().sum(dim=1).unsqueeze(1).expand_as(sum_out)
        text_len = text_len + text_len.eq(0).float()  # avoid divide zero denominator
        mean_out = sum_out.div(text_len)
        return mean_out

    def compute_mask(self, mask):
        B, N, T = mask.shape
        mask = mask.reshape(B * N, T)
        mask_sum = mask.sum(dim=-1)
        graph_node_mask = mask_sum != 0
        graph_node_mask = graph_node_mask.unsqueeze(-1).expand(B * N, T)
        src_key_padding_mask = torch.logical_not(mask.bool()) & graph_node_mask
        return src_key_padding_mask, graph_node_mask

    def forward(self, **kwargs):
        whole_image = kwargs['whole_image']
        text_segments = kwargs['text_segments']
        text_length = kwargs['text_length']
        iob_tags_label = kwargs['iob_tags_label'] if self.training else None
        mask = kwargs['mask']
        boxes_coordinate = kwargs['boxes_coordinate']

        text_emb = self.character_emb(text_segments)
        key_padding_mask, graph_node_mask = self.compute_mask(mask)
        character_embedding = self.encoder(whole_image, boxes_coordinate, text_emb, key_padding_mask)
        text_mask = torch.logical_not(key_padding_mask).byte()
        segment_embedding = self.aggregate(character_embedding, text_mask)
        graph_node_mask = graph_node_mask.any(dim=-1, keepdim=True)
        segment_embedding = segment_embedding * graph_node_mask.byte()

        B, N, T = mask.shape
        init_adj = torch.ones((B, N, N), device=text_emb.device)
        segment_embedding = self.graph(segment_embedding.reshape(B, N, -1), init_adj)
        logits, new_mask, log_likelihood = self.decoder(character_embedding.reshape(B, N, T, -1),
                                                        segment_embedding, mask, text_length, iob_tags_label)
        output = {"logits": logits, "new_mask": new_mask}

        if self.training:
            crf_loss = -log_likelihood
            output['crf_loss'] = crf_loss
        return output

    def __str__(self):
        '''
        Model prints with number of trainable parameters
        '''
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def model_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params