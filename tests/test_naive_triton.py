import pytest
import torch
from conftest import autograd_grads, requires_cuda

from flash_attention import attention_reference

pytestmark = requires_cuda

# (B, H, N, D): the original checks, plus non-power-of-two N and long sequences
SHAPES = [
    (1, 1, 32, 32),
    (1, 2, 64, 32),
    (2, 4, 64, 64),
    (2, 4, 128, 64),
    (2, 8, 256, 64),
    (1, 2, 100, 64),
    (1, 2, 333, 64),
    (1, 2, 256, 128),
    (1, 2, 512, 64),
    (1, 2, 1024, 64),
    (1, 1, 2048, 64),
    (1, 1, 4096, 64),
]


def _inputs(shape):
    torch.manual_seed(42)
    return (torch.randn(shape, device="cuda", dtype=torch.float32) for _ in range(4))


@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_forward(shape):
    from flash_attention import attention_forward_triton

    Q, K, V, _ = _inputs(shape)
    torch.testing.assert_close(attention_forward_triton(Q, K, V), attention_reference(Q, K, V), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_backward_matches_autograd(shape):
    from flash_attention import attention_backward_triton

    Q, K, V, dO = _inputs(shape)
    expected = autograd_grads(attention_reference, Q, K, V, dO)
    actual = attention_backward_triton(Q, K, V, dO)

    for name, a, e in zip(("dQ", "dK", "dV"), actual, expected):
        torch.testing.assert_close(a, e, atol=1e-4, rtol=1e-4, msg=name)


def test_unscaled():
    from flash_attention import attention_backward_triton, attention_forward_triton

    Q, K, V, dO = _inputs((1, 2, 128, 64))
    ref = lambda q, k, v: attention_reference(q, k, v, scale=1.0)  # noqa: E731
    torch.testing.assert_close(attention_forward_triton(Q, K, V, scale=1.0), ref(Q, K, V), atol=1e-4, rtol=1e-4)
    for a, e in zip(attention_backward_triton(Q, K, V, dO, scale=1.0), autograd_grads(ref, Q, K, V, dO)):
        torch.testing.assert_close(a, e, atol=1e-3, rtol=1e-3)
