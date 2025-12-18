# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron Module"""

import paddle

from megatron import get_args
from megatron.core import mpu, tensor_parallel


_FLOAT_TYPES = (paddle.float32,)
_HALF_TYPES = (paddle.float16,)
_BF16_TYPES = (paddle.bfloat16,)


def param_is_not_shared(param):
    return not hasattr(param, 'shared') or not param.shared



class MegatronModule(paddle.nn.Layer):
    """Megatron specific extensions of paddle Layer with support
    for pipelining."""

    def __init__(self, share_word_embeddings=True):
        super(MegatronModule, self).__init__()
        self.share_word_embeddings = share_word_embeddings


    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """Use this function to override the state dict for
        saving checkpoints."""
        return self.state_dict(structured_name_prefix=prefix, keep_vars=keep_vars)


    def word_embeddings_weight(self):
        if self.pre_process:
            return self.language_model.embedding.word_embeddings.weight
        else:
            if not self.share_word_embeddings:
                raise Exception('word_embeddings_weight() called for last '
                                'stage, but share_word_embeddings is false')
            return self.word_embeddings.weight


    def initialize_word_embeddings(self, init_method_normal):
        args = get_args()
        if not self.share_word_embeddings:
            raise Exception('initialize_word_embeddings() was called but '
                            'share_word_embeddings is false')

        # This function just initializes the word embeddings in the final stage
        # when we are using pipeline parallelism. Nothing to do if we aren't
        # using pipeline parallelism.
        if args.pipeline_model_parallel_size == 1:
            return

        # Parameters are shared between the word embeddings layers, and the
        # heads at the end of the model. In a pipelined setup with more than
        # one stage, the initial embedding layer and the head are on different
        # workers, so we do the following:
        # 1. Create a second copy of word_embeddings on the last stage, with
        #    initial parameters of 0.0.
        # 2. Do an all-reduce between the first and last stage to ensure that
        #    the two copies of word_embeddings start off with the same
        #    parameter values.
        # 3. In the training loop, before an all-reduce between the grads of
        #    the two word_embeddings layers to ensure that every applied weight
        #    update is the same on both stages.
        if mpu.is_pipeline_last_stage() and not self.pre_process:
            assert not mpu.is_pipeline_first_stage()
            self._word_embeddings_for_head_key = 'word_embeddings_for_head'
            # set word_embeddings weights to 0 here, then copy first
            # stage's weights using all_reduce below.
            self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
                args.padded_vocab_size, args.hidden_size,
                init_method=init_method_normal(args.init_method_std),
                params_dtype=args.params_dtype,
                use_cpu_initialization=args.use_cpu_initialization,
                perform_initialization=args.perform_initialization)
            self.word_embeddings.weight.data.fill_(0)
            self.word_embeddings.weight.shared = True

        # Zero out initial weights for decoder embedding.
        # NOTE: We don't currently support T5 with the interleaved schedule.
        if not mpu.is_pipeline_first_stage(ignore_virtual=True) and \
                self.pre_process:
            self.language_model.embedding.zero_parameters()

        if not paddle.distributed.is_initialized():
            if not getattr(MegatronModule, "embedding_warning_printed", False):
                print("WARNING! Distributed processes aren't initialized, so "
                      "word embeddings in the last layer are not initialized. "
                      "If you are just manipulating a model this is fine, but "
                      "this needs to be handled manually. If you are training "
                      "something is definitely wrong.")
                MegatronModule.embedding_warning_printed = True
            return

        # Ensure that first and last stages have the same initial parameter
        # values.
        if mpu.is_rank_in_embedding_group():
            paddle.distributed.all_reduce(self.word_embeddings_weight().data,
                                         group=mpu.get_embedding_group())

        # Ensure that encoder(first stage) and decoder(split stage) position
        # embeddings have the same initial parameter values
        # NOTE: We don't currently support T5 with the interleaved schedule.
        if mpu.is_rank_in_position_embedding_group() and \
                args.pipeline_model_parallel_split_rank is not None:
            # TODO: Support tokentype embedding.
            # 这里Paddle不需要进行CUDA的迁移
            # self.language_model.embedding.cuda()
            position_embeddings = self.language_model.embedding.position_embeddings
            paddle.distributed.all_reduce(position_embeddings.weight.data,
                                         group=mpu.get_position_embedding_group())


def conversion_helper(val, conversion):
    """Apply conversion to val. Recursively apply conversion if `val`
    #is a nested tuple/list structure."""
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn = [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        rtn = tuple(rtn)
    return rtn

# TODO: Check the fllowing code whether correct
def fp32_to_float16(val, float16_convertor):
    """Convert fp32 `val` to fp16/bf16"""
    def half_conversion(val):

        if isinstance(val, paddle.Tensor) and val.dtype == paddle.float32:
            val = float16_convertor(val)
        return val
    return conversion_helper(val, half_conversion)


def float16_to_fp32(val):
    """Convert fp16/bf16 `val` to fp32"""
    def float_conversion(val):
        if isinstance(val, paddle.Tensor) and val.dtype in (paddle.float16, paddle.bfloat16):
            val = val.astype(paddle.float32)
        return val
    return conversion_helper(val, float_conversion)



class Float16Module(MegatronModule):

    def __init__(self, module, args):
        super(Float16Module, self).__init__()

        if args.fp16:
            module = module.astype(paddle.float16)
            def float16_convertor(val):
                return val.astype(paddle.float16)
        elif args.bf16:

            module = module.astype(paddle.bfloat16)
            def float16_convertor(val):
                return val.astype(paddle.bfloat16)
        else:
            raise Exception('should not be here')

        self.add_sublayer('module', module)
        self.float16_convertor = float16_convertor


    def set_input_tensor(self, input_tensor):
        return self.module.set_input_tensor(input_tensor)


    def forward(self, *inputs, **kwargs):
        if mpu.is_pipeline_first_stage():
            inputs = fp32_to_float16(inputs, self.float16_convertor)
        outputs = self.module(*inputs, **kwargs)
        if mpu.is_pipeline_last_stage():
            outputs = float16_to_fp32(outputs)
        return outputs


    def state_dict(self):
        return self.module.state_dict()


    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        return self.module.state_dict_for_save_checkpoint(prefix=prefix,
                                                          keep_vars=keep_vars)


    def set_state_dict(self, state_dict):
        self.module.set_state_dict(state_dict)
