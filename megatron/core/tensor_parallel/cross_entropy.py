# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import paddle

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size
)

from .utils import VocabUtility


class _VocabParallelCrossEntropy(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, label_smoothing=0.0):
        # Maximum value along vocab dimension across all GPUs.
        logits_max = paddle.max(vocab_parallel_logits, axis=-1)
        paddle.distributed.all_reduce(
            logits_max,
            op=paddle.distributed.ReduceOp.MAX,
            group=get_tensor_model_parallel_group()
        )
        
        # Subtract the maximum value.
        vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(axis=-1)

        # Get the partition's vocab indices
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.shape[-1]
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(
            partition_vocab_size, rank, world_size)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target = paddle.where(target_mask, paddle.zeros_like(masked_target), masked_target)

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = paddle.reshape(vocab_parallel_logits, [-1, partition_vocab_size])
        masked_target_1d = paddle.reshape(masked_target, [-1])
        arange_1d = paddle.arange(0, logits_2d.shape[0], dtype='int64')
        predicted_logits_1d = paddle.gather_nd(
            logits_2d,
            paddle.stack([arange_1d, masked_target_1d], axis=1)
        )
        predicted_logits = paddle.reshape(predicted_logits_1d, target.shape)
        predicted_logits = paddle.where(target_mask, paddle.zeros_like(predicted_logits), predicted_logits)
        
        # All reduce is needed to get the chunks from other GPUs.
        paddle.distributed.all_reduce(
            predicted_logits,
            op=paddle.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group()
        )

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = paddle.exp(vocab_parallel_logits)
        sum_exp_logits = paddle.sum(exp_logits, axis=-1)
        paddle.distributed.all_reduce(
            sum_exp_logits,
            op=paddle.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group()
        )

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = paddle.log(sum_exp_logits) - predicted_logits

        # Normalize and optionally smooth logits
        exp_logits = exp_logits / sum_exp_logits.unsqueeze(axis=-1)

        vocab_size = exp_logits.shape[-1]
        if label_smoothing > 0:
            """
            We'd like to assign 1 / (K - 1) probability mass to every index that is not the ground truth.
            = (1 - alpha) * y_gt + alpha * mean(y_{i for i != gt})
            = (1 - alpha) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = ((K - 1) * (1 - alpha) / (K - 1)) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = (K * (1 - alpha) - 1) / (K - 1)) * y_gt  + (alpha / (K - 1)) * \sum_{i} y_i
            = (1 - (alpha * K) / (K - 1)) * y_gt + ( (alpha * K) / (K - 1) ) * \sum_{i} y_i / K
            """
            assert 1.0 > label_smoothing > 0.0
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)

            # Exp logits at this point are normalized probabilities. So we can just take the log to get log-probs.
            log_probs = paddle.log(exp_logits)
            mean_log_probs = paddle.mean(log_probs, axis=-1)
            loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

        ctx.label_smoothing = label_smoothing
        ctx.vocab_size = vocab_size
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors
        label_smoothing, vocab_size = ctx.label_smoothing, ctx.vocab_size

        # All the inputs have softmax as their gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.shape[-1]
        grad_2d = paddle.reshape(grad_input, [-1, partition_vocab_size])

        # Add the gradient from matching classes.
        arange_1d = paddle.arange(0, grad_2d.shape[0], dtype='int64')
        softmax_update = 1.0 - target_mask.astype('float32').reshape([-1])

        if label_smoothing > 0:
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)
            grad_2d_update = paddle.scatter_nd(
                paddle.stack([arange_1d, masked_target_1d], axis=1),
                -(1.0 - smoothing) * softmax_update,
                shape=grad_2d.shape
            )
            grad_2d = grad_2d + grad_2d_update
            average_grad = -smoothing / vocab_size
            grad_2d = grad_2d + average_grad
        else:
            grad_2d_update = paddle.scatter_nd(
                paddle.stack([arange_1d, masked_target_1d], axis=1),
                -softmax_update,
                shape=grad_2d.shape
            )
            grad_2d = grad_2d + grad_2d_update

        # Finally elementwise multiplication with the output gradients.
        grad_input = grad_input * grad_output.unsqueeze(axis=-1)

        return grad_input, None, None


def vocab_parallel_cross_entropy(vocab_parallel_logits, target, label_smoothing=0.0):
    """
    Performs cross entropy loss when logits are split across tensor parallel ranks

    Arguments:
        vocab_parallel_logits: logits split across tensor parallel ranks
                             dimension is [sequence_length, batch_size, hidden_size]
        target: correct vocab ids of dimension [sequence_length, micro_batch_size]
        label_smoothing: smoothing factor, must be in range [0.0, 1.0)
                        default is no smoothing (=0.0)
    """
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, label_smoothing)