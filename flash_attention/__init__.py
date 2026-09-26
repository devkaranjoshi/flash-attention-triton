"""Attention in Triton: a fused online-softmax forward, and a naive
multi-head forward/backward validated against PyTorch autograd."""

from .reference import attention_backward_reference, attention_reference

__all__ = [
    "attention_reference",
    "attention_backward_reference",
]

try:  # Triton kernels need a CUDA-capable Triton install
    from .naive_triton import attention_backward_triton, attention_forward_triton
    from .online_forward import flash_attention_forward

    __all__ += [
        "attention_forward_triton",
        "attention_backward_triton",
        "flash_attention_forward",
    ]
except ImportError:
    pass
