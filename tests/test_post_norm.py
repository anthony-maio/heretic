# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from heretic.model import get_norm_gain, trace_post_norm_gains

DIM = 32


class RMSNorm(nn.Module):
    def __init__(self, centered: bool = False) -> None:
        super().__init__()
        self.centered = centered
        self.weight = nn.Parameter(torch.randn(DIM) * 0.5)

    def forward(self, x: Tensor) -> Tensor:
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-8)
        return y * ((1 + self.weight) if self.centered else self.weight)


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.o_proj = nn.Linear(DIM, DIM, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.o_proj(x)


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_proj = nn.Linear(DIM, DIM, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(x)


class LlamaLayer(nn.Module):
    # Pre-norm, with the pre-MLP norm named "post_attention_layernorm".
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm()
        self.self_attn = Attention()
        self.post_attention_layernorm = RMSNorm()
        self.mlp = MLP()

    def forward(self, h: Tensor) -> Tensor:
        h = h + self.self_attn(self.input_layernorm(h))
        return h + self.mlp(self.post_attention_layernorm(h))


class GemmaLayer(nn.Module):
    # Sandwich norms: the block outputs are normalized before the residual add.
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(centered=True)
        self.self_attn = Attention()
        self.post_attention_layernorm = RMSNorm(centered=True)
        self.pre_feedforward_layernorm = RMSNorm(centered=True)
        self.mlp = MLP()
        self.post_feedforward_layernorm = RMSNorm(centered=True)

    def forward(self, h: Tensor) -> Tensor:
        h = h + self.post_attention_layernorm(self.self_attn(self.input_layernorm(h)))
        return h + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(h))
        )


class MisnamedPreNormLayer(nn.Module):
    # Pre-norm, but both norms are named "post_*" (ZGCM-1 does this).
    def __init__(self) -> None:
        super().__init__()
        self.post_attention_layernorm = RMSNorm()
        self.self_attn = Attention()
        self.post_feedforward_layernorm = RMSNorm()
        self.mlp = MLP()

    def forward(self, h: Tensor) -> Tensor:
        h = h + self.self_attn(self.post_attention_layernorm(h))
        return h + self.mlp(self.post_feedforward_layernorm(h))


def trace(layer_class: type[nn.Module]) -> tuple[nn.ModuleList, dict]:
    torch.manual_seed(0)
    layers = nn.ModuleList([layer_class() for _ in range(2)])
    layer_modules = [
        {
            "attn.o_proj": [layer.self_attn.o_proj],
            "mlp.down_proj": [layer.mlp.down_proj],
        }
        for layer in layers
    ]

    def run_forward() -> None:
        h = torch.randn(1, 3, DIM)
        for layer in layers:
            h = layer(h)

    return layers, trace_post_norm_gains(layers, layer_modules, run_forward)


class PostNormTests(unittest.TestCase):
    def test_pre_norm_llama_finds_nothing(self) -> None:
        _, gains = trace(LlamaLayer)
        self.assertEqual(gains, {})

    def test_misnamed_pre_norm_finds_nothing(self) -> None:
        _, gains = trace(MisnamedPreNormLayer)
        self.assertEqual(gains, {})

    def test_sandwich_norm_finds_every_module_with_its_gain(self) -> None:
        layers, gains = trace(GemmaLayer)
        self.assertEqual(len(gains), 4)
        for layer_index, layer in enumerate(layers):
            torch.testing.assert_close(
                gains[(layer_index, "attn.o_proj", 0)],
                1 + layer.post_attention_layernorm.weight.detach(),
            )
            torch.testing.assert_close(
                gains[(layer_index, "mlp.down_proj", 0)],
                1 + layer.post_feedforward_layernorm.weight.detach(),
            )

    def test_gain_probe_handles_layernorm_with_bias(self) -> None:
        torch.manual_seed(0)
        norm = nn.LayerNorm(DIM)
        nn.init.normal_(norm.weight)
        nn.init.normal_(norm.bias)
        torch.testing.assert_close(
            get_norm_gain(norm, DIM, torch.device("cpu")), norm.weight.detach()
        )

    def test_corrected_direction_leaves_nothing_after_the_norm(self) -> None:
        layers, gains = trace(GemmaLayer)
        layer = layers[0]
        W = layer.self_attn.o_proj.weight.detach()
        v = F.normalize(torch.randn(DIM), dim=0)
        x = torch.randn(64, DIM)
        norm = layer.post_attention_layernorm

        def leak(u: Tensor) -> float:
            # The ablation heretic applies: W - u u^T W.
            W_ablated = W - torch.outer(u, u @ W)
            return (norm(x @ W_ablated.T) @ v).abs().max().item()

        corrected = F.normalize(v * gains[(0, "attn.o_proj", 0)], dim=0)
        self.assertGreater(leak(v), 1e-2)
        self.assertLess(leak(corrected), 1e-5)


if __name__ == "__main__":
    unittest.main()
