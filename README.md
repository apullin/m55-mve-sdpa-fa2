# Int8 Tiled Attention Demo

This project is a small, self-contained reference for a 3-layer int8 attention stack over a fixed `2250 x 24` input sequence. The implementation keeps two full `2250 x 24` sequence buffers live, plus one full-sequence `K` cache and one full-sequence `V` cache for the active head. Attention is still evaluated in query/key tiles, so there is no full attention-score tensor in memory.

The current layout is:

- `2250 x 24` input sequence
- `3` transformer-style blocks
- attention widened internally to `32`
- `2` heads, `16` channels per head
- `24 -> 32 -> 24` attention path and `24 -> 24 -> 24` feed-forward per block
- RoPE applied to `Q` and `K` with a baked q15 sin/cos table
- `30` query rows per tile and `150` key rows per tile, so `2250` divides cleanly
- `2:1` average pooling at the end
- `1125 x 4` classifier output

CMSIS-DSP is consumed from the adjacent `../CMSIS-DSP` checkout. The projection GEMMs use `arm_mat_mult_q7`, the residual and bias paths use q7 vector primitives, and the score path uses `arm_dot_prod_q7` now that each head is `16` lanes wide. The runtime-saving trick is that each head's full `K` and `V` sequences are materialized once, reused across all query tiles, and then discarded before moving to the next head. Each head's context tile is projected through its own `W_o` slice immediately, so there is no full `2250 x 32` attention buffer either.

`tools/torch_reference.py` is a PyTorch reference that mirrors the fixed-point edge path, including q7 GEMM semantics, the q15 RoPE table, and the q12 LUT softmax. It intentionally does not use `torch.sdpa`, because standard SDPA is not bit-equivalent to the edge implementation.

## Build

Host build:

```sh
cmake -S . -B build
cmake --build build -j4
./build/tiled_attention
```

Cortex-M55 + MVE QEMU smoke build:

```sh
cmake -S . -B build-qemu \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-none-eabi-gcc.cmake \
  -DARM_M55_QEMU=ON \
  -DMODEL_INPUT_SEQ_LEN=128
cmake --build build-qemu -j4
cmake --build build-qemu --target run_qemu
```

For the full `2250`-token shape under QEMU, omit `-DMODEL_INPUT_SEQ_LEN=128`. That is useful for correctness, but it is much slower than the smoke profile.
