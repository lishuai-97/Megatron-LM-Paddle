# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Utilities for models."""

import math

import paddle
import paddle.nn as nn

from megatron import get_args

def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""
    def init_(tensor):
        return nn.initializer.Normal(mean=0.0, std=sigma)(tensor)
    return init_


def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers))."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return nn.initializer.Normal(mean=0.0, std=std)(tensor)
    return init_


def attention_mask_func(attention_scores, attention_mask):
    attention_scores = paddle.where(
        attention_mask,
        paddle.full_like(attention_scores, -10000.0),
        attention_scores
    )
    return attention_scores


def get_linear_layer(rows, columns, init_method):
    """Simple linear layer with weight initialization."""
    layer = nn.Linear(rows, columns)
    if get_args().perform_initialization:
        init_method(layer.weight)
    layer.bias.set_value(paddle.zeros_like(layer.bias))
    return layer

@paddle.jit.to_static
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + paddle.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))
def openai_gelu(x):
    return gelu_impl(x)

@paddle.jit.to_static
def erf_gelu(x):
    return x * 0.5 * (paddle.erf(x / 1.41421).astype(x.dtype) + paddle.ones_like(x).astype(x.dtype))
