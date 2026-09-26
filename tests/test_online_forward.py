import pytest
import torch
from conftest import requires_cuda

pytestmark = requires_cuda

# (N, D, BLOCK_M, BLOCK_N)
CONFIGS = [
    (64, 64, 16, 16),
    (128, 64, 32, 32),
    (256, 64, 64, 64),
    (1024, 64, 64, 64),
    (4096, 64, 64, 64),
    (100, 64, 32, 32),
    (1000, 64, 64, 64),
    (512, 32, 16, 16),
    (256, 128, 32, 32),
]


@pytest.mark.parametrize("N,D,BLOCK_M,BLOCK_N", CONFIGS)
def test_online_forward_matches_softmax_attention(N, D, BLOCK_M, BLOCK_N):
    from flash_attention import flash_attention_forward

    torch.manual_seed(0)
    Q, K, V = (torch.randn(N, D, device="cuda") for _ in range(3))

    out = flash_attention_forward(Q, K, V, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    ref = torch.softmax(Q @ K.T, dim=-1) @ V  # kernel applies no 1/sqrt(D) scale

    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
