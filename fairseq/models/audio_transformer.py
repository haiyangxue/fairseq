# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable

from fairseq import options, utils
import argparse

from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    AudioEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
    TransformerAudioDecoderLayer,
    VGGBlock,
)
import random
from ..data.data_utils import lengths_to_encoder_padding_mask


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model('audio_transformer')
class AudioTransformerModel(AudioEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    @classmethod
    def hub_models(cls):
        # fmt: off
        return {
            'transformer.wmt14.en-fr': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2',
            'transformer.wmt16.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2',
            'transformer.wmt18.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz',
            'transformer.wmt19.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz',
            'transformer.wmt19.en-ru': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz',
            'transformer.wmt19.de-en': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz',
            'transformer.wmt19.ru-en': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz',
            'transformer.wmt19.en-de.single_model': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz',
            'transformer.wmt19.en-ru.single_model': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.single_model.tar.gz',
            'transformer.wmt19.de-en.single_model': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.single_model.tar.gz',
            'transformer.wmt19.ru-en.single_model': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.single_model.tar.gz',
        }
        # fmt: on

    def __init__(self, encoder,audio_encoder, decoder):
        super().__init__(encoder,audio_encoder, decoder)
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        parser.add_argument('--layer-wise-attention', default=False, action='store_true',
                            help='perform layer-wise attention (cross-attention or cross+self-attention)')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)

        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        audio_encoder = cls.build_audio_encoder(args)

        # encoder=None
        # return cls(encoder, decoder)
        #
        return cls(encoder,audio_encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_audio_encoder(cls, args):
        return VGGTransformerEncoder(
            input_feat_per_channel=args.input_feat_per_channel,
            vggblock_config=eval(args.vggblock_enc_config),
            transformer_config=eval(args.transformer_enc_config),
            encoder_output_dim=args.enc_output_dim,
            in_channels=args.in_channels,
        )
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, 'no_cross_attention', False),
        )

def prepare_transformer_encoder_params(input_dim, num_heads,ffn_dim,normalize_before,dropout,attention_dropout,relu_dropout,
):
    args = argparse.Namespace()
    args.encoder_embed_dim = input_dim
    args.encoder_attention_heads = num_heads
    # args.attention_dropout = attention_dropout
    # args.dropout = dropout
    # args.activation_dropout = relu_dropout
    args.attention_dropout = 0
    args.dropout = 0
    args.activation_dropout = 0
    args.encoder_normalize_before = normalize_before
    args.encoder_ffn_embed_dim = ffn_dim

    return args

class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout
        self.encoder_layerdrop = args.encoder_layerdrop
        self.src_dic=dictionary
        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        def freeze_module_params(self):
            if self is not None:
                for p in self.parameters():
                    p.requires_grad = False

        freeze_module_params(self)

    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def forward(self, src_tokens, src_lengths, cls_input=None, return_all_hiddens=False):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False)

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        if self.layer_wise_attention:
            return_all_hiddens = True
        x, encoder_embedding = self.forward_embedding(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.encoder_layerdrop):
                x = layer(x, encoder_padding_mask)
                if return_all_hiddens:
                    encoder_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)
            if return_all_hiddens:
                encoder_states[-1] = x
        # print(encoder_padding_mask)
        # exit()
        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
            'encoder_embedding': encoder_embedding,  # B x T x C
            'encoder_states': encoder_states,  # List[T x B x C]
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        if encoder_out.get('encoder_states', None) is not None:
            for idx, state in enumerate(encoder_out['encoder_states']):
                encoder_out['encoder_states'][idx] = state.index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
            if self._future_mask.size(0) < dim:
                self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        for i in range(len(self.layers)):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(state_dict, "{}.layers.{}".format(name, i))

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict

class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))
        self.tgt_dic=dictionary
        self.dropout = args.dropout
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, self.padding_idx,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.cross_self_attention = getattr(args, 'cross_self_attention', False)
        self.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

        self.layers = nn.ModuleList([])

        self.layers.extend([
            TransformerAudioDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])
        # self.layers.extend([
        #     TransformerDecoderLayer(args, no_encoder_attn)
        #     for _ in range(args.decoder_layers)
        # ])
        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, self.output_embed_dim, bias=False) \
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), self.output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if args.decoder_normalize_before and not getattr(args, 'no_decoder_final_norm', False):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(
        self,
        prev_output_tokens,
        encoder_out=None,
        audio_encoder_out=None,
        incremental_state=None,
        features_only=False,
        **extra_args,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        # exit()
        x, extra = self.extract_features(
            prev_output_tokens, encoder_out,audio_encoder_out, incremental_state, **extra_args,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, None

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        audio_encoder_out=None,
        incremental_state=None,
        full_context_alignment=False,
        alignment_layer=None,
        alignment_heads=None,
        **unused,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = len(self.layers) - 1

        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)
        if not self_attn_padding_mask.any() and not self.cross_self_attention:
            self_attn_padding_mask = None

        # decoder layers
        attn = None
        inner_states = [x]
        # print( encoder_out['encoder_padding_mask'])
        # exit()
        for idx, layer in enumerate(self.layers):
            encoder_state = None
            if encoder_out is not None:
                if self.layer_wise_attention:
                    encoder_state = encoder_out['encoder_states'][idx]
                else:
                    encoder_state = encoder_out['encoder_out']
            audio_encoder_state=None
            if audio_encoder_out is not None:
                audio_encoder_state = audio_encoder_out['encoder_out']
                # print(audio_encoder_state.size())
                audio_encoder_out['encoder_padding_mask']=None
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.decoder_layerdrop):
                x, layer_attn = layer(
                    x,
                    encoder_state,
                    encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                    audio_encoder_state,
                    audio_encoder_out['encoder_padding_mask'] if audio_encoder_out is not None else None,
                    incremental_state=incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=(idx == alignment_layer),
                    need_head_weights=(idx == alignment_layer),
                )
                inner_states.append(x)
                if layer_attn is not None and idx == alignment_layer:
                    attn = layer_attn.float()

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {'attn': attn, 'inner_states': inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight)
            else:
                return F.linear(features, self.embed_out)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, '_future_mask')
            or self._future_mask is None
            or self._future_mask.device != tensor.device
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


DEFAULT_ENC_VGGBLOCK_CONFIG = ((32, 3, 2, 2, False),) * 2
DEFAULT_ENC_TRANSFORMER_CONFIG = ((256, 4, 1024, True, 0.2, 0.2, 0.2),) * 2
# 256: embedding dimension
# 4: number of heads
# 1024: FFN
# True: apply layerNorm before (dropout + resiaul) instead of after
# 0.2 (dropout): dropout after MultiheadAttention and second FC
# 0.2 (attention_dropout): dropout in MultiheadAttention
# 0.2 (relu_dropout): dropout after ReLu
DEFAULT_DEC_TRANSFORMER_CONFIG = ((256, 2, 1024, True, 0.2, 0.2, 0.2),) * 2
DEFAULT_DEC_CONV_CONFIG = ((256, 3, True),) * 2

class VGGTransformerEncoder(FairseqEncoder):
    """VGG + Transformer encoder"""

    def __init__(
            self,
            input_feat_per_channel,
            vggblock_config=DEFAULT_ENC_VGGBLOCK_CONFIG,
            transformer_config=DEFAULT_ENC_TRANSFORMER_CONFIG,
            encoder_output_dim=512,
            in_channels=1,
            transformer_context=None,
            transformer_sampling=None,
    ):
        """constructor for VGGTransformerEncoder

        Args:
            - input_feat_per_channel: feature dim (not including stacked,
              just base feature)
            - in_channel: # input channels (e.g., if stack 8 feature vector
                together, this is 8)
            - vggblock_config: configuration of vggblock, see comments on
                DEFAULT_ENC_VGGBLOCK_CONFIG
            - transformer_config: configuration of transformer layer, see comments
                on DEFAULT_ENC_TRANSFORMER_CONFIG
            - encoder_output_dim: final transformer output embedding dimension
            - transformer_context: (left, right) if set, self-attention will be focused
              on (t-left, t+right)
            - transformer_sampling: an iterable of int, must match with
              len(transformer_config), transformer_sampling[i] indicates sampling
              factor for i-th transformer layer, after multihead att and feedfoward
              part
        """
        super().__init__(None)

        self.num_vggblocks = 0
        if vggblock_config is not None:
            if not isinstance(vggblock_config, Iterable):
                raise ValueError("vggblock_config is not iterable")
            self.num_vggblocks = len(vggblock_config)

        self.conv_layers = nn.ModuleList()
        self.in_channels = in_channels
        self.input_dim = input_feat_per_channel

        if vggblock_config is not None:
            for _, config in enumerate(vggblock_config):
                (out_channels,conv_kernel_size,pooling_kernel_size,num_conv_layers,layer_norm,) = config
                self.conv_layers.append(
                    VGGBlock(in_channels,out_channels,conv_kernel_size,pooling_kernel_size,num_conv_layers,
                        input_dim=input_feat_per_channel,layer_norm=layer_norm,)
                )
                in_channels = out_channels
                input_feat_per_channel = self.conv_layers[-1].output_dim

        transformer_input_dim = self.infer_conv_output_dim(
            self.in_channels, self.input_dim
        )
        # transformer_input_dim is the output dimension of VGG part

        self.validate_transformer_config(transformer_config)
        self.transformer_context = self.parse_transformer_context(transformer_context)
        self.transformer_sampling = self.parse_transformer_sampling(
            transformer_sampling, len(transformer_config)
        )

        self.transformer_layers = nn.ModuleList()

        if transformer_input_dim != transformer_config[0][0]:
            self.transformer_layers.append(Linear(transformer_input_dim, transformer_config[0][0]))
        self.transformer_layers.append(
            TransformerEncoderLayer(prepare_transformer_encoder_params(*transformer_config[0]))
        )
        for i in range(1, len(transformer_config)):
            if transformer_config[i - 1][0] != transformer_config[i][0]:
                self.transformer_layers.append(Linear(transformer_config[i - 1][0], transformer_config[i][0]))
            self.transformer_layers.append(
                TransformerEncoderLayer(prepare_transformer_encoder_params(*transformer_config[i])))
        self.encoder_output_dim = encoder_output_dim
        self.transformer_layers.extend(
            [Linear(transformer_config[-1][0], encoder_output_dim),
                LayerNorm(encoder_output_dim), ]
        )
        def freeze_module_params(self):
            # for n,p in self.named_parameters():
            #     print(n)
            #     if n=="transformer_layers.1.fc1.weight":
            #         print(p)
            #         exit()
            if self is not None:
                for p in self.parameters():
                    p.requires_grad = False

        freeze_module_params(self)
        # exit()

    def forward(self, src_tokens, src_lengths, **kwargs):
        """
        src_tokens: padded tensor (B, T, C * feat)
        src_lengths: tensor of original lengths of input utterances (B,)
        """
        # for n, p in self.named_parameters():
        #     print(n)
        #     print(p)
        #
        #     if n == "transformer_layers.0.weight":
        #         exit()
        # print(src_tokens)
        # print(src_tokens.size())
        # print(src_lengths)
        bsz, max_seq_len, _ = src_tokens.size()
        x = src_tokens.view(bsz, max_seq_len, self.in_channels, self.input_dim)
        x = x.transpose(1, 2).contiguous()
        # print("x_1")
        # print(x)
        # (B, C, T, feat)
        for layer_idx in range(len(self.conv_layers)):
            x = self.conv_layers[layer_idx](x)
            # print("*****")
            # print(x)
        bsz, _, output_seq_len, _ = x.size()
        # (B, C, T, feat) -> (B, T, C, feat) -> (T, B, C, feat) -> (T, B, C * feat)
        x = x.transpose(1, 2).transpose(0, 1)
        x = x.contiguous().view(output_seq_len, bsz, -1)
        # print("x_2")
        # print(x)
        # exit()
        subsampling_factor = int(max_seq_len * 1.0 / output_seq_len + 0.5)

        # print("subsampling........")
        # print(output_seq_len)
        # print(max_seq_len)
        # print(subsampling_factor)
        # print(src_lengths.float())
        # TODO: shouldn't subsampling_factor determined in advance ?
        # print(src_lengths)
        input_lengths = (src_lengths.float() / subsampling_factor).ceil().long()
        # print(input_lengths)
        encoder_padding_mask, _ = lengths_to_encoder_padding_mask(
            input_lengths, batch_first=True
        )
        # print("encoder_padding_mask")
        # print(encoder_padding_mask)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None
        # encoder_padding_mask = None

        attn_mask = self.lengths_to_attn_mask(input_lengths, subsampling_factor)
        # print("attn_mask")
        # print(attn_mask)
        transformer_layer_idx = 0
        for layer_idx in range(len(self.transformer_layers)):
            # print(layer_idx)
            # print(x)
            if isinstance(self.transformer_layers[layer_idx], TransformerEncoderLayer):
                # print("***************")
                # print(x.size())
                # if encoder_padding_mask is  not None:
                #     print(encoder_padding_mask.size())
                # else:
                #     print(encoder_padding_mask)


                # print(encoder_padding_mask)
                # print(layer_idx)
                # print("x")
                # print(x)
                # print("encoder_padding_mask")
                #
                # print(encoder_padding_mask)
                # print("attn_mask")
                #
                # print(attn_mask)
                x = self.transformer_layers[layer_idx](
                    x, encoder_padding_mask, attn_mask)
                # print("after x")
                # print(x)
                if self.transformer_sampling[transformer_layer_idx] != 1:
                    sampling_factor = self.transformer_sampling[transformer_layer_idx]
                    x, encoder_padding_mask, attn_mask = self.slice(
                        x, encoder_padding_mask, attn_mask, sampling_factor)
                transformer_layer_idx += 1

            else:
                x = self.transformer_layers[layer_idx](x)
            # print("encoder out"+str(layer_idx))
            # print(x)
        # encoder_padding_maks is a (T x B) tensor, its [t, b] elements indicate
        # whether encoder_output[t, b] is valid or not (valid=0, invalid=1)
        # print("encoder out")
        # print(x)
        return {
            "encoder_out": x,  # (T, B, C)
            "encoder_padding_mask": encoder_padding_mask.t()
            if encoder_padding_mask is not None
            else None,
            # (B, T) --> (T, B)
        }

    def infer_conv_output_dim(self, in_channels, input_dim):
        sample_seq_len = 200
        sample_bsz = 10
        x = torch.randn(sample_bsz, in_channels, sample_seq_len, input_dim)
        for i, _ in enumerate(self.conv_layers):
            x = self.conv_layers[i](x)
        x = x.transpose(1, 2)
        mb, seq = x.size()[:2]
        return x.contiguous().view(mb, seq, -1).size(-1)

    def validate_transformer_config(self, transformer_config):
        for config in transformer_config:
            input_dim, num_heads = config[:2]
            if input_dim % num_heads != 0:
                msg = (
                        "ERROR in transformer config {}:".format(config)
                        + "input dimension {} ".format(input_dim)
                        + "not dividable by number of heads".format(num_heads)
                )
                raise ValueError(msg)

    def parse_transformer_context(self, transformer_context):
        """
        transformer_context can be the following:
        -   None; indicates no context is used, i.e.,
            transformer can access full context
        -   a tuple/list of two int; indicates left and right context,
            any number <0 indicates infinite context
                * e.g., (5, 6) indicates that for query at x_t, transformer can
                access [t-5, t+6] (inclusive)
                * e.g., (-1, 6) indicates that for query at x_t, transformer can
                access [0, t+6] (inclusive)
        """
        if transformer_context is None:
            return None

        if not isinstance(transformer_context, Iterable):
            raise ValueError("transformer context must be Iterable if it is not None")

        if len(transformer_context) != 2:
            raise ValueError("transformer context must have length 2")

        left_context = transformer_context[0]
        if left_context < 0:
            left_context = None

        right_context = transformer_context[1]
        if right_context < 0:
            right_context = None

        if left_context is None and right_context is None:
            return None

        return (left_context, right_context)

    def parse_transformer_sampling(self, transformer_sampling, num_layers):
        """
        parsing transformer sampling configuration
        Args:
            - transformer_sampling, accepted input:
                * None, indicating no sampling
                * an Iterable with int (>0) as element
            - num_layers, expected number of transformer layers, must match with
              the length of transformer_sampling if it is not None
        Returns:
            - A tuple with length num_layers
        """
        if transformer_sampling is None:
            return (1,) * num_layers
        if not isinstance(transformer_sampling, Iterable):
            raise ValueError(
                "transformer_sampling must be an iterable if it is not None"
            )
        if len(transformer_sampling) != num_layers:
            raise ValueError(
                "transformer_sampling {} does not match with the number "
                + "of layers {}".format(transformer_sampling, num_layers)
            )
        for layer, value in enumerate(transformer_sampling):
            if not isinstance(value, int):
                raise ValueError("Invalid value in transformer_sampling: ")
            if value < 1:
                raise ValueError(
                    "{} layer's subsampling is {}.".format(layer, value)
                    + " This is not allowed! "
                )
        return transformer_sampling

    def slice(self, embedding, padding_mask, attn_mask, sampling_factor):
        """
        embedding is a (T, B, D) tensor
        padding_mask is a (B, T) tensor or None
        attn_mask is a (T, T) tensor or None
        """
        embedding = embedding[::sampling_factor, :, :]
        if padding_mask is not None:
            padding_mask = padding_mask[:, ::sampling_factor]
        if attn_mask is not None:
            attn_mask = attn_mask[::sampling_factor, ::sampling_factor]

        return embedding, padding_mask, attn_mask

    def lengths_to_attn_mask(self, input_lengths, subsampling_factor=1):
        """
        create attention mask according to sequence lengths and transformer
        context
        Args:
            - input_lengths: (B, )-shape Int/Long tensor; input_lengths[b] is
              the length of b-th sequence
            - subsampling_factor: int
                * Note that the left_context and right_context is specified in
                  the input frame-level while input to transformer may already
                  go through subsampling (e.g., the use of striding in vggblock)
                  we use subsampling_factor to scale the left/right context
        Return:
            - a (T, T) binary tensor or None, where T is max(input_lengths)
                * if self.transformer_context is None, None
                * if left_context is None,
                    * attn_mask[t, t + right_context + 1:] = 1
                    * others = 0
                * if right_context is None,
                    * attn_mask[t, 0:t - left_context] = 1
                    * others = 0
                * elsif
                    * attn_mask[t, t - left_context: t + right_context + 1] = 0
                    * others = 1
        """
        if self.transformer_context is None:
            return None
        maxT = torch.max(input_lengths).item()
        attn_mask = torch.zeros(maxT, maxT)

        left_context = self.transformer_context[0]
        right_context = self.transformer_context[1]
        if left_context is not None:
            left_context = math.ceil(self.transformer_context[0] / subsampling_factor)
        if right_context is not None:
            right_context = math.ceil(self.transformer_context[1] / subsampling_factor)

        for t in range(maxT):
            if left_context is not None:
                st = 0
                en = max(st, t - left_context)
                attn_mask[t, st:en] = 1
            if right_context is not None:
                st = t + right_context + 1
                st = min(st, maxT - 1)
                attn_mask[t, st:] = 1

        return attn_mask.to(input_lengths.device)

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
            1, new_order
        )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(1, new_order)
        return encoder_out


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

def base_audio(args):
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 40)
    # args.vggblock_enc_config = getattr(
    #     args, "vggblock_enc_config", DEFAULT_ENC_VGGBLOCK_CONFIG
    # )
    # args.transformer_enc_config = getattr(
    #     args, "transformer_enc_config", DEFAULT_ENC_TRANSFORMER_CONFIG
    # )
    args.enc_output_dim = getattr(args, "enc_output_dim", 512)
    args.in_channels = getattr(args, "in_channels", 1)
    args.tgt_embed_dim = getattr(args, "tgt_embed_dim", 128)
    args.transformer_context = getattr(args, "transformer_context", "None")

def vggtransformer_2(args):

    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.vggblock_enc_config = getattr(
        args, "vggblock_enc_config", "[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]"
    )
    args.transformer_enc_config = getattr(
        args,
        "transformer_enc_config",
        "((1024, 16, 4096, True, 0.15, 0.15, 0.15),) * 16",
    )
    args.enc_output_dim = getattr(args, "enc_output_dim", 1024)
    base_audio(args)



@register_model_architecture('audio_transformer', 'audio_transformer')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.no_cross_attention = getattr(args, 'no_cross_attention', False)
    args.cross_self_attention = getattr(args, 'cross_self_attention', False)
    args.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)
    vggtransformer_2(args)


@register_model_architecture('audio_transformer', 'audio_transformer_iwslt_de_en')
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    base_architecture(args)


@register_model_architecture('audio_transformer', 'audio_transformer_wmt_en_de')
def transformer_wmt_en_de(args):
    base_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture('audio_transformer', 'audio_transformer_vaswani_wmt_en_de_big')
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.dropout = getattr(args, 'dropout', 0.3)
    base_architecture(args)


@register_model_architecture('audio_transformer', 'audio_transformer_vaswani_wmt_en_fr_big')
def transformer_vaswani_wmt_en_fr_big(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture('audio_transformer', 'audio_transformer_wmt_en_de_big')
def transformer_wmt_en_de_big(args):
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture('audio_transformer', 'audio_transformer_wmt_en_de_big_t2t')
def transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)
#
#
# @register_model_architecture('transformer_align', 'transformer_align')
# def transformer_align(args):
#     args.alignment_heads = getattr(args, 'alignment_heads', 1)
#     args.alignment_layer = getattr(args, 'alignment_layer', 4)
#     args.full_context_alignment = getattr(args, 'full_context_alignment', False)
#     base_architecture(args)
#
#
# @register_model_architecture('transformer_align', 'transformer_wmt_en_de_big_align')
# def transformer_wmt_en_de_big_align(args):
#     args.alignment_heads = getattr(args, 'alignment_heads', 1)
#     args.alignment_layer = getattr(args, 'alignment_layer', 4)
#     transformer_wmt_en_de_big(args)
