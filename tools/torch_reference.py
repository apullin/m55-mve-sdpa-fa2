#!/usr/bin/env python3
"""PyTorch reference for the int8 tiled-attention edge path.

This intentionally mirrors the C implementation's fixed-point dataflow:
- q7 GEMM semantics for the linear layers
- q15 RoPE tables
- streamed online softmax with the same q12 LUT
- q7 residual / bias saturation

It does not use torch.sdpa. Standard SDPA is useful for training, but it is not
bit-equivalent to the edge implementation because the edge path uses fixed-point
projections, a LUT softmax, and online renormalization.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

import gen_model_data as modelgen


@dataclass(frozen=True)
class ModelConfig:
    model_dim: int = 24
    attn_dim: int = 32
    num_heads: int = 2
    num_layers: int = 3
    input_seq_len: int = 1200
    num_classes: int = 4
    query_tile: int = 30
    key_tile: int = 150

    @property
    def head_dim(self) -> int:
        return self.attn_dim // self.num_heads

class XorShift32:
    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF
        if self.state == 0:
            self.state = 0x6D2B79F5

    def next_u32(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x & 0xFFFFFFFF
        return self.state


def sat_q7(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-128, 127).to(torch.int32)


def add_q7(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return sat_q7(a.to(torch.int32) + b.to(torch.int32))


def as_int_tensor(data: object) -> torch.Tensor:
    return torch.tensor(data, dtype=torch.int32)


def build_demo_input(cfg: ModelConfig) -> torch.Tensor:
    sequence = torch.empty((cfg.input_seq_len, cfg.model_dim), dtype=torch.int32)
    rng = XorShift32(0x1A2B3C4D)

    for token in range(cfg.input_seq_len):
        for feature in range(cfg.model_dim):
            trend = ((token * 3 + feature * 11) % 29) - 14
            stripe = 7 if ((token + feature) & 1) else -7
            noise = (rng.next_u32() & 0x0F) - 8
            sequence[token, feature] = max(-128, min(127, trend + stripe + noise))

    return sequence


class ReferenceLayer(nn.Module):
    def __init__(self, layer_data: dict[str, list]) -> None:
        super().__init__()
        self.register_buffer("w_q", as_int_tensor(layer_data["w_q"]))
        self.register_buffer("w_k", as_int_tensor(layer_data["w_k"]))
        self.register_buffer("w_v", as_int_tensor(layer_data["w_v"]))
        self.register_buffer("w_o", as_int_tensor(layer_data["w_o"]))
        self.register_buffer("b_o", as_int_tensor(layer_data["b_o"]))
        self.register_buffer("w_ff1", as_int_tensor(layer_data["w_ff1"]))
        self.register_buffer("b_ff1", as_int_tensor(layer_data["b_ff1"]))
        self.register_buffer("w_ff2", as_int_tensor(layer_data["w_ff2"]))
        self.register_buffer("b_ff2", as_int_tensor(layer_data["b_ff2"]))


class EdgeReferenceModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.attn_dim % cfg.num_heads != 0:
            raise ValueError("attn_dim must be divisible by num_heads")
        if cfg.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        self.cfg = cfg
        self.layers = nn.ModuleList(
            ReferenceLayer(
                modelgen.generate_layer(0x12345678 + 0x10203 * layer_idx, cfg.model_dim, cfg.num_heads, cfg.head_dim)
            )
            for layer_idx in range(cfg.num_layers)
        )

        classifier_rng = modelgen.XorShift32(0xCAFEBABE)
        classifier_w = modelgen.make_tensor(
            classifier_rng, (cfg.model_dim, cfg.num_classes), magnitude=10, zero_mask=0x3
        )
        classifier_b = modelgen.make_tensor(classifier_rng, (cfg.num_classes,), magnitude=3, zero_mask=0x1)
        rope_cos_q15, rope_sin_q15 = modelgen.build_rope_tables(cfg.input_seq_len, cfg.head_dim)

        self.register_buffer("classifier_w", as_int_tensor(classifier_w))
        self.register_buffer("classifier_b", as_int_tensor(classifier_b))
        self.register_buffer("softmax_lut", as_int_tensor(modelgen.build_softmax_lut()))
        self.register_buffer("rope_cos_q15", as_int_tensor(rope_cos_q15))
        self.register_buffer("rope_sin_q15", as_int_tensor(rope_sin_q15))

    def q7_linear_block(self, input_block: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        acc = input_block.to(torch.int32) @ weights.to(torch.int32)
        out = sat_q7(acc >> 7)
        if bias is None:
            return out
        return add_q7(out, bias.view(1, -1).expand_as(out))

    def apply_rope_q7(self, data: torch.Tensor, start_pos: int) -> torch.Tensor:
        rows = data.shape[0]
        pair_dim = self.cfg.head_dim // 2
        pairs = data.to(torch.int32).reshape(rows, pair_dim, 2)
        cos_q15 = self.rope_cos_q15[start_pos : start_pos + rows]
        sin_q15 = self.rope_sin_q15[start_pos : start_pos + rows]

        x0 = pairs[:, :, 0]
        x1 = pairs[:, :, 1]
        round_q15 = 1 << (modelgen.ROPE_Q15_SHIFT - 1)
        y0 = (x0 * cos_q15 - x1 * sin_q15 + round_q15) >> modelgen.ROPE_Q15_SHIFT
        y1 = (x0 * sin_q15 + x1 * cos_q15 + round_q15) >> modelgen.ROPE_Q15_SHIFT

        out = torch.empty_like(pairs)
        out[:, :, 0] = sat_q7(y0)
        out[:, :, 1] = sat_q7(y1)
        return out.reshape(rows, self.cfg.head_dim)

    def softmax_decay_q12(self, delta: torch.Tensor) -> torch.Tensor:
        delta = delta.to(torch.int64)
        lut_size = int(self.softmax_lut.numel() - 1)
        out = torch.zeros_like(delta, dtype=torch.int32)

        non_pos = delta <= 0
        in_range = (delta > 0) & (delta < lut_size)

        out[non_pos] = modelgen.SOFTMAX_Q12_ONE
        out[in_range] = self.softmax_lut[delta[in_range].to(torch.long)]
        return out

    def build_head_kv_cache(self, sequence: torch.Tensor, layer: ReferenceLayer, head_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        k_cache = self.q7_linear_block(sequence, layer.w_k[head_idx], None)
        k_cache = self.apply_rope_q7(k_cache, 0)
        v_cache = self.q7_linear_block(sequence, layer.w_v[head_idx], None)
        return k_cache, v_cache

    def run_attention_head_block_with_cache(
        self,
        sequence: torch.Tensor,
        layer: ReferenceLayer,
        head_idx: int,
        query_start: int,
        query_rows: int,
        head_k_cache: torch.Tensor,
        head_v_cache: torch.Tensor,
    ) -> torch.Tensor:
        query_tile = self.q7_linear_block(
            sequence[query_start : query_start + query_rows], layer.w_q[head_idx], None
        )
        query_tile = self.apply_rope_q7(query_tile, query_start)

        row_max = torch.full((query_rows,), -(1 << 15), dtype=torch.int32)
        row_sum = torch.zeros((query_rows,), dtype=torch.int32)
        row_acc = torch.zeros((query_rows, self.cfg.head_dim), dtype=torch.int32)
        round_q12 = modelgen.SOFTMAX_Q12_ONE // 2

        for key_start in range(0, self.cfg.input_seq_len, self.cfg.key_tile):
            key_end = min(key_start + self.cfg.key_tile, self.cfg.input_seq_len)
            key_tile = head_k_cache[key_start:key_end]
            value_tile = head_v_cache[key_start:key_end]
            scores = (query_tile.to(torch.int32) @ key_tile.to(torch.int32).transpose(0, 1)) >> modelgen.ATTN_SCORE_SHIFT

            for key_idx in range(key_end - key_start):
                score = scores[:, key_idx]
                value_vec = value_tile[key_idx].to(torch.int32)
                first_mask = row_sum == 0

                if torch.any(first_mask):
                    row_max[first_mask] = score[first_mask]
                    row_sum[first_mask] = modelgen.SOFTMAX_Q12_ONE
                    row_acc[first_mask] = modelgen.SOFTMAX_Q12_ONE * value_vec

                active_mask = ~first_mask
                if not torch.any(active_mask):
                    continue

                grow_mask = active_mask & (score > row_max)
                if torch.any(grow_mask):
                    alpha_q12 = self.softmax_decay_q12(score[grow_mask] - row_max[grow_mask])
                    row_sum[grow_mask] = ((row_sum[grow_mask] * alpha_q12 + round_q12) >> 12).to(torch.int32)
                    row_acc[grow_mask] = (
                        (row_acc[grow_mask] * alpha_q12.unsqueeze(1) + round_q12) >> 12
                    ).to(torch.int32)
                    row_max[grow_mask] = score[grow_mask]
                    row_sum[grow_mask] += modelgen.SOFTMAX_Q12_ONE
                    row_acc[grow_mask] += modelgen.SOFTMAX_Q12_ONE * value_vec

                stay_mask = active_mask & ~grow_mask
                if torch.any(stay_mask):
                    weight_q12 = self.softmax_decay_q12(row_max[stay_mask] - score[stay_mask])
                    row_sum[stay_mask] += weight_q12
                    row_acc[stay_mask] += weight_q12.unsqueeze(1) * value_vec

        denom = row_sum.unsqueeze(1)
        pos_rounded = torch.div(row_acc + denom // 2, denom.clamp_min(1), rounding_mode="trunc")
        neg_rounded = torch.div(row_acc - denom // 2, denom.clamp_min(1), rounding_mode="trunc")
        rounded = torch.where(row_acc >= 0, pos_rounded, neg_rounded)
        rounded = torch.where(denom > 0, rounded, torch.zeros_like(rounded))
        return sat_q7(rounded)

    def run_transformer_layer(self, current_sequence: torch.Tensor, layer: ReferenceLayer) -> torch.Tensor:
        next_sequence = torch.zeros_like(current_sequence)

        for head_idx in range(self.cfg.num_heads):
            head_k_cache, head_v_cache = self.build_head_kv_cache(current_sequence, layer, head_idx)

            for query_start in range(0, self.cfg.input_seq_len, self.cfg.query_tile):
                query_rows = min(self.cfg.query_tile, self.cfg.input_seq_len - query_start)
                head_context = self.run_attention_head_block_with_cache(
                    current_sequence,
                    layer,
                    head_idx,
                    query_start,
                    query_rows,
                    head_k_cache,
                    head_v_cache,
                )
                projected = self.q7_linear_block(
                    head_context,
                    layer.w_o[head_idx * self.cfg.head_dim : (head_idx + 1) * self.cfg.head_dim],
                    None,
                )
                next_sequence[query_start : query_start + query_rows] = add_q7(
                    next_sequence[query_start : query_start + query_rows], projected
                )

        for query_start in range(0, self.cfg.input_seq_len, self.cfg.query_tile):
            query_rows = min(self.cfg.query_tile, self.cfg.input_seq_len - query_start)
            attention_out = add_q7(
                next_sequence[query_start : query_start + query_rows],
                layer.b_o.view(1, -1).expand(query_rows, -1),
            )
            residual = add_q7(current_sequence[query_start : query_start + query_rows], attention_out)
            work = self.q7_linear_block(residual, layer.w_ff1, layer.b_ff1)
            work = work.clamp(0, 127)
            context = self.q7_linear_block(work, layer.w_ff2, layer.b_ff2)
            next_sequence[query_start : query_start + query_rows] = add_q7(residual, context)

        return next_sequence

    def classify_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        output = torch.empty((self.cfg.input_seq_len, self.cfg.num_classes), dtype=torch.int32)

        for output_start in range(0, self.cfg.input_seq_len, self.cfg.query_tile):
            output_rows = min(self.cfg.query_tile, self.cfg.input_seq_len - output_start)
            output[output_start : output_start + output_rows] = self.q7_linear_block(
                sequence[output_start : output_start + output_rows], self.classifier_w, self.classifier_b
            )

        return output

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        current = sat_q7(sequence)
        for layer in self.layers:
            current = self.run_transformer_layer(current, layer)
        return self.classify_sequence(current).to(torch.int8)


def checksum_q7(data: torch.Tensor) -> int:
    checksum = 0
    flat = data.to(torch.int32).reshape(-1)
    for value in flat:
        checksum = (checksum * 131 + (int(value) & 0xFF)) & 0xFFFFFFFF
    return checksum


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-seq-len", type=int, default=128, help="Sequence length for the demo run")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = ModelConfig(input_seq_len=args.input_seq_len)
    model = EdgeReferenceModel(cfg)
    sequence = build_demo_input(cfg)
    output = model(sequence)

    preview_rows = min(8, output.shape[0])
    for row in range(preview_rows):
        values = " ".join(f"{int(v):4d}" for v in output[row].to(torch.int32))
        print(f"{row:4d}: {values}")
    print(f"checksum: 0x{checksum_q7(output):08x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
