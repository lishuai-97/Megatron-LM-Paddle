"""Transformer based language model."""

from typing import Optional, Tuple, Union, Dict, List
import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from megatron.core import tensor_parallel
from .enums import LayerType, AttnMaskType
from .module import MegatronModule
from .transformer import ParallelTransformer
from .utils import get_linear_layer
from .utils import init_method_normal, scaled_init_method_normal


def parallel_lm_logits(
    input_: paddle.Tensor,
    word_embeddings_weight: paddle.Tensor,
    parallel_output: bool,
    bias: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    """LM logits using word embedding weights.
    
    Args:
        input_: Input tensor [s, b, h]
        word_embeddings_weight: Weight tensor [v, h]
        parallel_output: Whether to keep output in model parallel form
        bias: Optional bias tensor [v]
        
    Returns:
        Logits tensor [s, b, v] or [s, b, v/p]
    """
    input_parallel = input_
    
    # Matrix multiply
    logits_parallel = tensor_parallel.linear_with_grad_accumulation(
        input=input_parallel,
        weight=word_embeddings_weight,
        bias=bias)

    if parallel_output:
        return logits_parallel

    return tensor_parallel.gather_from_model_parallel_region(logits_parallel)


class Embedding(MegatronModule):
    """Language model embeddings.
    
    Args:
        hidden_size: Hidden size
        vocab_size: Vocabulary size
        max_sequence_length: Maximum sequence length
        embedding_dropout_prob: Dropout probability for embeddings
        init_method: Weight initialization method
        num_tokentypes: Size of token-type embeddings (default: 0)
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        max_sequence_length: int,
        embedding_dropout_prob: float,
        init_method,
        num_tokentypes: int = 0
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.init_method = init_method
        self.num_tokentypes = num_tokentypes

        # Word embeddings
        self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
            vocab_size, self.hidden_size,
            weight_attr=paddle.ParamAttr(initializer=self.init_method))
        self._word_embeddings_key = 'word_embeddings'

        # Position embedding
        self.position_embeddings = nn.Embedding(
            max_sequence_length, self.hidden_size,
            weight_attr=paddle.ParamAttr(initializer=self.init_method))
        self._position_embeddings_key = 'position_embeddings'

        # Token type embedding
        self._tokentype_embeddings_key = 'tokentype_embeddings'
        if self.num_tokentypes > 0:
            self.tokentype_embeddings = nn.Embedding(
                self.num_tokentypes, self.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=self.init_method))
        else:
            self.tokentype_embeddings = None

        # Embeddings dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout_prob)

    def forward(
        self,
        input_ids: paddle.Tensor,
        position_ids: paddle.Tensor,
        tokentype_ids: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        # Word embeddings
        words_embeddings = self.word_embeddings(input_ids)
        
        # Position embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings

        # Token type embeddings
        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            embeddings = embeddings + self.tokentype_embeddings(tokentype_ids)
        else:
            assert self.tokentype_embeddings is None

        # [b s h] -> [s b h]
        embeddings = paddle.transpose(embeddings, perm=[1, 0, 2])

        # Dropout
        embeddings = self.embedding_dropout(embeddings)

        return embeddings


class TransformerLanguageModel(MegatronModule):
    """Transformer language model.
    
    Args:
        init_method: Initialization method for weights
        output_layer_init_method: Initialization method for output layer
        encoder_attn_mask_type: Attention mask type for encoder
        num_tokentypes: Number of token types (default: 0) 
        add_encoder: Whether to add encoder (default: True)
        add_decoder: Whether to add decoder (default: False)
        decoder_attn_mask_type: Attention mask type for decoder (default: causal)
        add_pooler: Whether to add pooler (default: False)
        pre_process: Whether to do preprocessing (default: True)
        post_process: Whether to do postprocessing (default: True)
    """

    def __init__(
        self,
        init_method,
        output_layer_init_method,
        encoder_attn_mask_type: AttnMaskType,
        num_tokentypes: int = 0,
        add_encoder: bool = True,
        add_decoder: bool = False,
        decoder_attn_mask_type: AttnMaskType = AttnMaskType.causal,
        add_pooler: bool = False,
        pre_process: bool = True,
        post_process: bool = True
    ):
        super().__init__()

        self.pre_process = pre_process
        self.post_process = post_process
        self.hidden_size = 1024  # TODO: Make configurable
        self.num_tokentypes = num_tokentypes
        self.init_method = init_method
        self.add_encoder = add_encoder
        self.encoder_attn_mask_type = encoder_attn_mask_type
        self.add_decoder = add_decoder
        self.decoder_attn_mask_type = decoder_attn_mask_type
        self.add_pooler = add_pooler
        self.encoder_hidden_state = None

        # Embeddings
        if self.pre_process:
            self.embedding = Embedding(
                self.hidden_size,
                vocab_size=32000,  # TODO: Make configurable
                max_sequence_length=2048,  # TODO: Make configurable
                embedding_dropout_prob=0.1,  # TODO: Make configurable
                init_method=self.init_method,
                num_tokentypes=self.num_tokentypes)
            self._embedding_key = 'embedding'

        # Encoder
        if self.add_encoder:
            self.encoder = ParallelTransformer(
                init_method=self.init_method,
                output_layer_init_method=output_layer_init_method,
                self_attn_mask_type=self.encoder_attn_mask_type,
                pre_process=self.pre_process,
                post_process=self.post_process)
            self._encoder_key = 'encoder'
        else:
            self.encoder = None

        # Decoder
        if self.add_decoder:
            self.decoder = ParallelTransformer(
                init_method=self.init_method,
                output_layer_init_method=output_layer_init_method,
                layer_type=LayerType.decoder,
                self_attn_mask_type=self.decoder_attn_mask_type,
                pre_process=self.pre_process,
                post_process=self.post_process)
            self._decoder_key = 'decoder'
        else:
            self.decoder = None

        if self.post_process:
            # Pooler
            if self.add_pooler:
                self.pooler = Pooler(self.hidden_size, self.init_method)
                self._pooler_key = 'pooler'

    def forward(
        self,
        enc_input_ids: paddle.Tensor,
        enc_position_ids: paddle.Tensor,
        enc_attn_mask: paddle.Tensor,
        dec_input_ids: Optional[paddle.Tensor] = None,
        dec_position_ids: Optional[paddle.Tensor] = None,
        dec_attn_mask: Optional[paddle.Tensor] = None,
        enc_dec_attn_mask: Optional[paddle.Tensor] = None,
        tokentype_ids: Optional[paddle.Tensor] = None,
        inference_params=None,
        pooling_sequence_index: int = 0,
        enc_hidden_states: Optional[paddle.Tensor] = None,
        output_enc_hidden: bool = False
    ) -> Union[paddle.Tensor, Tuple[paddle.Tensor, ...]]:
        # Encoder embedding
        if self.pre_process:
            encoder_input = self.embedding(
                enc_input_ids,
                enc_position_ids,
                tokentype_ids=tokentype_ids)
        else:
            encoder_input = None

        # Run encoder
        if enc_hidden_states is None:
            if self.encoder is not None:
                encoder_output = self.encoder(
                    encoder_input,
                    enc_attn_mask,
                    inference_params=inference_params)
            else:
                encoder_output = self.encoder_hidden_state
        else:
            encoder_output = enc_hidden_states.astype(encoder_input.dtype)

        if self.post_process:
            if self.add_pooler:
                pooled_output = self.pooler(
                    encoder_output,
                    pooling_sequence_index)

        # Return encoder output if needed
        if not self.add_decoder or output_enc_hidden:
            if self.add_pooler and self.post_process:
                return encoder_output, pooled_output
            else:
                return encoder_output

        # Decoder embedding
        if self.pre_process:
            decoder_input = self.embedding(
                dec_input_ids,
                dec_position_ids)
        else:
            decoder_input = None

        # Run decoder
        decoder_output = self.decoder(
            decoder_input,
            dec_attn_mask,
            encoder_output=encoder_output,
            enc_dec_attn_mask=enc_dec_attn_mask,
            inference_params=inference_params)

        if self.add_pooler and self.post_process:
            return decoder_output, encoder_output, pooled_output
        else:
            return decoder_output, encoder_output