#!/usr/bin/env python3
"""Generate deterministic int8 weights and the fixed-point softmax LUT."""

from __future__ import annotations

import argparse
import math
import pathlib
import sys


SOFTMAX_Q12_ONE = 4096
SOFTMAX_LUT_SIZE = 64
ATTN_SCORE_SHIFT = 8
SOFTMAX_STEP = 0.25
ROPE_Q15_SHIFT = 15
ROPE_Q15_ONE = 1 << ROPE_Q15_SHIFT
ROPE_BASE = 10000.0
RMS_NORM_WEIGHT_SHIFT = 14
RMS_NORM_WEIGHT_ONE = 1 << RMS_NORM_WEIGHT_SHIFT
RMS_NORM_INPUT_SHIFT = 7
RMS_NORM_INPUT_ONE = 1 << RMS_NORM_INPUT_SHIFT
RMS_NORM_MEAN_Q31_SHIFT = 31 - 2 * RMS_NORM_INPUT_SHIFT
RMS_NORM_RMS_SHIFT = 8
RMS_NORM_INV_SHIFT = 16


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

    def next_q7(self, magnitude: int, zero_mask: int) -> int:
        word = self.next_u32()
        if zero_mask and (word & zero_mask) == 0:
            return 0
        return int(word % (2 * magnitude + 1)) - magnitude


def c_format(value: object, indent: int = 0) -> str:
    if not isinstance(value, list):
        return str(value)

    if not value:
        return "{}"

    if not isinstance(value[0], list):
        return "{ " + ", ".join(str(item) for item in value) + " }"

    pad = " " * indent
    child_pad = " " * (indent + 4)
    inner = ",\n".join(child_pad + c_format(item, indent + 4) for item in value)
    return "{\n" + inner + "\n" + pad + "}"


def make_tensor(rng: XorShift32, shape: tuple[int, ...], magnitude: int, zero_mask: int) -> list:
    if len(shape) == 1:
        return [rng.next_q7(magnitude, zero_mask) for _ in range(shape[0])]
    return [make_tensor(rng, shape[1:], magnitude, zero_mask) for _ in range(shape[0])]


def make_rms_weight_q14(rng: XorShift32, length: int, spread: int = 2048) -> list[int]:
    weights = []
    for _ in range(length):
        delta = int(rng.next_u32() % (2 * spread + 1)) - spread
        weights.append(max(-32768, min(32767, RMS_NORM_WEIGHT_ONE + delta)))
    return weights


def generate_layer(seed: int, model_dim: int, num_heads: int, head_dim: int, ffn_dim: int) -> dict[str, list]:
    rng = XorShift32(seed)
    attn_dim = num_heads * head_dim
    return {
        "rms_attn_w_q14": make_rms_weight_q14(rng, model_dim),
        "w_q": make_tensor(rng, (num_heads, model_dim, head_dim), magnitude=6, zero_mask=0x3),
        "w_k": make_tensor(rng, (num_heads, model_dim, head_dim), magnitude=6, zero_mask=0x3),
        "w_v": make_tensor(rng, (num_heads, model_dim, head_dim), magnitude=7, zero_mask=0x7),
        "w_o": make_tensor(rng, (attn_dim, model_dim), magnitude=8, zero_mask=0x7),
        "b_o": make_tensor(rng, (model_dim,), magnitude=2, zero_mask=0x1),
        "rms_ffn_w_q14": make_rms_weight_q14(rng, model_dim),
        "w_ff1": make_tensor(rng, (model_dim, ffn_dim), magnitude=9, zero_mask=0x7),
        "b_ff1": make_tensor(rng, (ffn_dim,), magnitude=2, zero_mask=0x1),
        "w_ff2": make_tensor(rng, (ffn_dim, model_dim), magnitude=8, zero_mask=0x7),
        "b_ff2": make_tensor(rng, (model_dim,), magnitude=2, zero_mask=0x1),
    }


def render_layer_initializer(layer: dict[str, list], indent: int = 8) -> str:
    pad = " " * indent
    parts = []
    for key in ("rms_attn_w_q14", "w_q", "w_k", "w_v", "w_o", "b_o", "rms_ffn_w_q14", "w_ff1", "b_ff1", "w_ff2", "b_ff2"):
        parts.append(f"{pad}.{key} = {c_format(layer[key], indent)},")
    return "\n".join(parts)


def build_softmax_lut() -> list[int]:
    lut = []
    for idx in range(SOFTMAX_LUT_SIZE + 1):
        value = int(round(math.exp(-SOFTMAX_STEP * idx) * SOFTMAX_Q12_ONE))
        lut.append(max(0, min(SOFTMAX_Q12_ONE, value)))
    return lut


def build_rope_tables(input_seq_len: int, head_dim: int) -> tuple[list[list[int]], list[list[int]]]:
    pair_dim = head_dim // 2
    cos_table: list[list[int]] = []
    sin_table: list[list[int]] = []

    for pos in range(input_seq_len):
        cos_row = []
        sin_row = []
        for pair_idx in range(pair_dim):
            angle = pos / (ROPE_BASE ** (2.0 * pair_idx / head_dim))
            cos_value = int(round(math.cos(angle) * ROPE_Q15_ONE))
            sin_value = int(round(math.sin(angle) * ROPE_Q15_ONE))
            cos_row.append(max(-32768, min(32767, cos_value)))
            sin_row.append(max(-32768, min(32767, sin_value)))
        cos_table.append(cos_row)
        sin_table.append(sin_row)

    return cos_table, sin_table


def render_header(
    model_dim: int,
    attn_dim: int,
    ffn_dim: int,
    num_heads: int,
    num_layers: int,
    input_seq_len: int,
    num_classes: int,
    query_tile: int,
    key_tile: int,
) -> str:
    if attn_dim % num_heads != 0:
        raise ValueError("attn_dim must be divisible by num_heads")

    head_dim = attn_dim // num_heads
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    layers = [
        generate_layer(0x12345678 + 0x10203 * layer_idx, model_dim, num_heads, head_dim, ffn_dim)
        for layer_idx in range(num_layers)
    ]
    classifier_rng = XorShift32(0xCAFEBABE)
    classifier_w = make_tensor(classifier_rng, (model_dim, num_classes), magnitude=10, zero_mask=0x3)
    classifier_b = make_tensor(classifier_rng, (num_classes,), magnitude=3, zero_mask=0x1)
    softmax_lut = build_softmax_lut()
    rope_cos_q15, rope_sin_q15 = build_rope_tables(input_seq_len, head_dim)

    layer_blocks = ",\n".join(
        "    {\n" + render_layer_initializer(layer, indent=8) + "\n    }" for layer in layers
    )

    return f"""#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include "arm_math.h"

#define MODEL_DIM {model_dim}
#define ATTN_DIM {attn_dim}
#define FFN_DIM {ffn_dim}
#define NUM_HEADS {num_heads}
#define HEAD_DIM {head_dim}
#define NUM_LAYERS {num_layers}
#define INPUT_SEQ_LEN {input_seq_len}
#define OUTPUT_SEQ_LEN INPUT_SEQ_LEN
#define NUM_CLASSES {num_classes}
#define QUERY_TILE {query_tile}
#define KEY_TILE {key_tile}
#define SOFTMAX_Q12_ONE {SOFTMAX_Q12_ONE}
#define SOFTMAX_LUT_SIZE {SOFTMAX_LUT_SIZE}
#define ATTN_SCORE_SHIFT {ATTN_SCORE_SHIFT}
#define ROPE_Q15_SHIFT {ROPE_Q15_SHIFT}
#define RMS_NORM_WEIGHT_SHIFT {RMS_NORM_WEIGHT_SHIFT}
#define RMS_NORM_WEIGHT_ONE {RMS_NORM_WEIGHT_ONE}
#define RMS_NORM_INPUT_SHIFT {RMS_NORM_INPUT_SHIFT}
#define RMS_NORM_INPUT_ONE {RMS_NORM_INPUT_ONE}
#define RMS_NORM_MEAN_Q31_SHIFT {RMS_NORM_MEAN_Q31_SHIFT}
#define RMS_NORM_RMS_SHIFT {RMS_NORM_RMS_SHIFT}
#define RMS_NORM_INV_SHIFT {RMS_NORM_INV_SHIFT}
#define ROPE_PAIR_DIM (HEAD_DIM / 2)

typedef struct
{{
    int16_t rms_attn_w_q14[MODEL_DIM];
    q7_t w_q[NUM_HEADS][MODEL_DIM][HEAD_DIM];
    q7_t w_k[NUM_HEADS][MODEL_DIM][HEAD_DIM];
    q7_t w_v[NUM_HEADS][MODEL_DIM][HEAD_DIM];
    q7_t w_o[ATTN_DIM][MODEL_DIM];
    q7_t b_o[MODEL_DIM];
    int16_t rms_ffn_w_q14[MODEL_DIM];
    q7_t w_ff1[MODEL_DIM][FFN_DIM];
    q7_t b_ff1[FFN_DIM];
    q7_t w_ff2[FFN_DIM][MODEL_DIM];
    q7_t b_ff2[MODEL_DIM];
}} model_layer_t;

static const model_layer_t g_model_layers[NUM_LAYERS] = {{
{layer_blocks}
}};

static const q7_t g_classifier_w[MODEL_DIM][NUM_CLASSES] = {c_format(classifier_w, 4)};
static const q7_t g_classifier_b[NUM_CLASSES] = {c_format(classifier_b, 4)};
static const uint16_t g_softmax_exp_q12[SOFTMAX_LUT_SIZE + 1] = {c_format(softmax_lut, 4)};
static const int16_t g_rope_cos_q15[INPUT_SEQ_LEN][ROPE_PAIR_DIM] = {c_format(rope_cos_q15, 4)};
static const int16_t g_rope_sin_q15[INPUT_SEQ_LEN][ROPE_PAIR_DIM] = {c_format(rope_sin_q15, 4)};

#endif
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_header")
    parser.add_argument("--model-dim", type=int, default=24)
    parser.add_argument("--attn-dim", type=int, default=32)
    parser.add_argument("--ffn-dim", type=int, default=48)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--input-seq-len", type=int, default=1200)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--query-tile", type=int, default=30)
    parser.add_argument("--key-tile", type=int, default=150)
    args = parser.parse_args()

    output_path = pathlib.Path(args.output_header)
    output_path.write_text(
        render_header(
            model_dim=args.model_dim,
            attn_dim=args.attn_dim,
            ffn_dim=args.ffn_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            input_seq_len=args.input_seq_len,
            num_classes=args.num_classes,
            query_tile=args.query_tile,
            key_tile=args.key_tile,
        ),
        encoding="ascii",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
