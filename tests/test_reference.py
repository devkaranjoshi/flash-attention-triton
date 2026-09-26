import pytest
import torch
from conftest import autograd_grads

from flash_attention import attention_backward_reference, attention_reference

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(2, 2, 8, 16), (1, 4, 33, 32), (2, 3, 64, 64)])
def test_manual_backward_matches_autograd(device, causal, shape):
    torch.manual_seed(0)
    Q, K, V, dO = (torch.randn(shape, device=device, dtype=torch.float64) for _ in range(4))

    expected = autograd_grads(lambda q, k, v: attention_reference(q, k, v, causal=causal), Q, K, V, dO)
    actual = attention_backward_reference(Q, K, V, dO, causal=causal)

    for name, a, e in zip(("dQ", "dK", "dV"), actual, expected):
        torch.testing.assert_close(a, e, atol=1e-10, rtol=1e-8, msg=name)


def test_scale_argument():
    torch.manual_seed(0)
    Q, K, V, dO = (torch.randn(1, 1, 16, 8, dtype=torch.float64) for _ in range(4))
    expected = autograd_grads(lambda q, k, v: attention_reference(q, k, v, scale=1.0), Q, K, V, dO)
    actual = attention_backward_reference(Q, K, V, dO, scale=1.0)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e)
