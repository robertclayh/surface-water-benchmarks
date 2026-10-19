import torch
from torch import nn

from swbench.cost.profile import forward_flops, inference_benchmark, train_step_flops
from swbench.models.vit import SimpleViTEncoder, ViTSegmenter


def test_conv_flops_exact():
    m = nn.Conv2d(3, 4, 1, bias=False)
    # 2 * Cin * Cout * H * W
    assert forward_flops(m, 3, size=8) == 2 * 3 * 4 * 64


def test_frozen_training_is_cheaper_than_finetune():
    torch.manual_seed(0)
    enc = SimpleViTEncoder(6, embed_dim=64, depth=4, heads=2, patch_size=16)
    frozen = ViTSegmenter(enc, decoder_channels=16, frozen=True)
    ft = ViTSegmenter(SimpleViTEncoder(6, embed_dim=64, depth=4, heads=2, patch_size=16), decoder_channels=16)
    f_fwd = forward_flops(frozen, 6, size=64)
    assert train_step_flops(frozen, 6, size=64) < train_step_flops(ft, 6, size=64)
    assert train_step_flops(ft, 6, size=64) > 2 * f_fwd


def test_inference_benchmark_cpu():
    out = inference_benchmark(nn.Conv2d(6, 2, 1), 6, torch.device("cpu"), batch_size=2, size=32, warmup=1, iters=2)
    assert out["infer_tiles_per_s"] > 0 and out["infer_km2_per_s"] > 0


def test_attention_flops_are_counted():
    import torch.nn.functional as F

    class Attn(nn.Module):
        def forward(self, x):
            q = x.reshape(1, 4, 49, 16)
            return F.scaled_dot_product_attention(q, q, q)

    # QK^T and AV: 2 matmuls x 2 FLOPs/MAC x L^2 x d x heads
    assert forward_flops(Attn(), 4, size=28) == 4 * 49 * 49 * 16 * 4
