# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Utility functions used throughout Megatron core"""
from functools import reduce
import operator

import paddle

from megatron.core import parallel_state


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def get_attr_wrapped_model(model, attr):
    """Get an attribute from a wrapped model"""
    if isinstance(model, list):
        raise RuntimeError("_get_attr_wrapped_model given a list of models")

    while not hasattr(model, attr):
        if not hasattr(model, "_layers"):
            raise RuntimeError(f"_get_attr_wrapped_model couldn't find attribute {attr}")
        model = model._layers

    return getattr(model, attr)


def get_model_type(model):
    return get_attr_wrapped_model(model, 'model_type')


class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.
    Caller should ensure that buffers of the same name
    are not used concurrently."""

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name):
        required_len = reduce(operator.mul, tensor_shape, 1)
        if self.buffer.get((name, dtype), None) is None or \
                self.buffer[(name, dtype)].numel() < required_len:
            self.buffer[(name, dtype)] = \
                paddle.empty(shape=[required_len],
                           dtype=dtype)

        return paddle.reshape(self.buffer[(name, dtype)][:required_len], tensor_shape)


def _kernel_make_viewless_tensor(inp, requires_grad):
    '''Make a viewless tensor.

    Create a new tensor that shares the same memory as the input tensor
    but doesn't maintain gradient history.
    '''
    # 创建参数
    out = paddle.create_parameter(
        shape=inp.shape,
        dtype=inp.dtype,
        default_initializer=paddle.nn.initializer.Assign(
            paddle.zeros(inp.shape, dtype=inp.dtype)
        )
    )
    
    # 设置stop_gradient状态
    out.stop_gradient = not requires_grad
    
    # 设置值
    out.set_value(inp)
    return out


class MakeViewlessTensor(paddle.autograd.PyLayer):
    '''
    Autograd function to make a viewless tensor.

    This function should be used in cases where the computation graph needs
    to be propagated, but we only want a viewless tensor.
    '''
    @staticmethod
    def forward(ctx, inp):  # 移除requires_grad参数
        # 从输入tensor的stop_gradient状态确定requires_grad
        requires_grad = not inp.stop_gradient
        out = _kernel_make_viewless_tensor(inp, requires_grad)
        return out

    @staticmethod
    def backward(ctx, grad_output):  # 只返回一个梯度
        return grad_output  # 直接返回梯度


def make_viewless_tensor(inp, requires_grad, keep_graph):
    '''
    Entry-point for creating viewless tensors.

    This method should be used, rather than calling 'MakeViewlessTensor'
    or '_kernel_make_viewless_tensor' directly.
    '''
    # 设置输入tensor的stop_gradient状态
    inp.stop_gradient = not requires_grad
    
    if keep_graph:
        out = MakeViewlessTensor.apply(inp)  # 不传递requires_grad参数
    else:
        out = _kernel_make_viewless_tensor(inp, requires_grad)
    
    # 确保输出tensor的stop_gradient状态正确
    out.stop_gradient = not requires_grad
    return out


def assert_viewless_tensor(tensor, extra_msg=None):
    '''Assert that a tensor is not a view.'''
    if isinstance(tensor, list):
        [assert_viewless_tensor(t) for t in tensor]
        return tensor
    if not isinstance(tensor, paddle.Tensor):
        return tensor
    return tensor


def safely_set_viewless_tensor_data(tensor, new_data_tensor):
    '''Safely set tensor's data.'''
    assert_viewless_tensor(tensor)
    tensor.set_value(new_data_tensor)