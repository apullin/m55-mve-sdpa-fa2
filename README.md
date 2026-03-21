# M55 MVE SDPA / FA2 Reference

This repo is a small, self-contained reference for low-memory int8 scaled dot-product attention on Cortex-M55 with Helium/MVE and CMSIS-DSP. The code is intentionally written as one straight-through C program in `src/main.c`: fixed sizes, fixed buffers, imperative control flow, and comments at the points where the dataflow or fixed-point math is not obvious.

The current target model is:

- input: `1200 x 24`
- output: `1200 x 4`
- blocks: `3`
- external model width: `24`
- internal attention width: `32`
- feed-forward hidden width: `48`
- heads: `2`
- head width: `16`
- attention path per block: `24 -> 32 -> 24`
- feed-forward path per block: `24 -> 48 -> 24`
- normalization: pre-RMSNorm before attention and before FFN
- position encoding: RoPE on `Q` and `K` with baked q15 sin/cos tables
- tile sizes: `QUERY_TILE=30`, `KEY_TILE=150`

This is full attention, not local attention and not architectural chunking. Every query position still attends to every key position. The tiling is only there to control memory traffic and keep scratch small.

## Repo Layout

- `src/main.c`: the edge-oriented C reference with CMSIS-DSP and MVE hot paths
- `tools/gen_model_data.py`: deterministic int8 parameter generator plus q12 softmax LUT and q15 RoPE table generator
- `tools/torch_reference.py`: PyTorch mirror of the fixed-point edge path
- `cmake/toolchains/arm-none-eabi-gcc.cmake`: Cortex-M55 cross-build setup

The FFN width is configurable at build time with `-DMODEL_FFN_DIM=...` without changing the low-memory execution schedule.

## Shapes And Dataflow

Each transformer block runs in two phases:

1. Per layer and per head:
   build one shared attention-normalized sequence for the layer
   project `Q`, `K`, `V`
   apply RoPE to `Q` and `K`
   compute tiled score blocks `Q_tile @ K_tile^T`
   run streamed online softmax over each score tile
   accumulate weighted `V`
   project the head context through that head's `W_o` slice and add it into the shared `24`-wide output buffer

2. Per tile:
   add output bias
   add the residual
   apply pre-RMSNorm to the residual
   run the widened `24 -> 48 -> 24` feed-forward block
   write the final tile back to the spare sequence buffer

After the last block, the model applies one direct `24 -> 4` classifier row by row.

## Memory Strategy

The code is designed around peak-memory control rather than maximum host speed.

The important tricks are:

- the full working set now lives in one runtime context allocation, so firmware can reserve it only for the duration of inference
- only two full `1200 x 24` sequence buffers are live at once
- one full `1200 x 24` attention-normalized sequence is materialized per layer so both heads can reuse it
- only one head's full `K` cache and one head's full `V` cache are materialized at a time
- FFN-side RMSNorm is still applied tile-by-tile, so there is no second full-sequence normalized buffer
- there is no full `Q`, `K`, `V`, attention-score, or probability tensor for the whole sequence
- each head context tile is projected through its `W_o` slice immediately, so there is no full `1200 x 32` attention output buffer
- score tiles are stored as compact `int16` scratch and consumed immediately by the online softmax/value pass

With the current default tiles, the score scratch is `30 x 150 x int16`, about `9 KB`.

## CMSIS-DSP And MVE Use

CMSIS-DSP is taken from the adjacent `../CMSIS-DSP` checkout.

The split is:

- CMSIS-DSP q7 GEMMs for projections and classifier via `arm_mat_mult_q7`
- CMSIS-DSP q7 vector primitives for residual, bias, clip, fill, and copy
- custom MVE intrinsics for the attention-specific hot path where CMSIS-DSP has no direct primitive

The custom MVE pieces in `src/main.c` now include:

- score microkernel for tiled `Q @ K^T`
- weighted-`V` accumulation
- q12 accumulator rescale
- q15 RoPE rotation
- sign-aware normalization setup before the final scalar divide

The final divide in normalization is still scalar because MVE does not provide vector integer divide.

## Exactness

`tools/torch_reference.py` mirrors the fixed-point edge path instead of using `torch.sdpa`. That is intentional: standard SDPA is not bit-equivalent to this implementation because this repo uses:

- q7 GEMM semantics
- q14 RMSNorm weights with fixed-point RMS evaluation, q31 sqrt, and q7 requantization
- q15 RoPE tables
- q12 LUT-based softmax decay
- online renormalized softmax accumulation
- saturating q7 residual and output paths

The PyTorch script is there to preserve semantic equivalence with the edge path, not to imitate a standard float attention implementation. It also exposes a Python-only dropout knob for experimentation, but the default parity path keeps dropout disabled.

## Parity Test

The repo includes a host-side parity harness that compares the full q7 output tensor from the C binary against the PyTorch fixed-point mirror.

1. Build the host binary.
2. Run `tools/test_torch_parity.py` with a Python interpreter that has `torch` installed.
3. By default the harness requires exact equality, but `--max-abs-diff` can be raised if you later want a tolerance-based check.

Example:

```sh
cmake --build build-host-1200-nopool -j4
/Users/andrewpullin/anaconda3/bin/python3.12 tools/test_torch_parity.py \
  --binary build-host-1200-nopool/tiled_attention \
  --python /Users/andrewpullin/anaconda3/bin/python3.12 \
  --input-seq-len 1200
```

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

For the full `1200`-token shape under QEMU, omit `-DMODEL_INPUT_SEQ_LEN=128`. QEMU is useful for correctness and smoke testing, not for meaningful M55 timing.

The bare-metal QEMU linker script reserves heap for that one-shot context allocation, matching the intended on-device lifetime model.

## QEMU Baseline Compare

The repo also includes a relative regression harness for comparing the current tree against the last committed baseline under QEMU:

```sh
/Users/andrewpullin/anaconda3/bin/python3.12 tools/compare_qemu_baseline.py \
  --input-seq-len 128 \
  --runs 3
```

What it does:

- builds host and QEMU binaries for the current tree
- builds host and QEMU binaries for baseline commit `1d2347c`
- checks host checksum == QEMU checksum for each variant
- reports repeated QEMU wall-time as a relative-only proxy

Use `--input-seq-len 1200` when you want the deployed shape instead of the smoke shape.

Important caveat: QEMU timing here is only for comparing two binaries against each other. It is not a real M55 timing number.

## Verifying MVE Survived Integration

If you integrate this code into a larger firmware tree, the two easiest checks are:

- runtime banner: the program prints one line of build info before inference:
  `mve-build: compiler_mve=... cmsis_mvei=... autovec_disabled=... probe=...`
- disassembly probe: `src/main.c` keeps a `used,noinline` helper named `tiled_attention_mve_probe_dot_16`

For a real M55+Helium build, you want:

- `compiler_mve=1`
- `cmsis_mvei=1`
- `autovec_disabled=1`

If any of those are wrong, the build is not on the intended MVE path.

The `ARM_REQUIRE_MVE` CMake option defaults to `ON` for Cortex-M55 builds in this repo. That adds a compile-time guard so the build fails if `__ARM_FEATURE_MVE` or `ARM_MATH_MVEI` is missing.

To inspect the probe in a cross-built binary:

```sh
arm-none-eabi-objdump -d build-qemu/tiled_attention | rg -n "tiled_attention_mve_probe_dot_16|vmlava|vldrb"
```

In the probe body, look for Helium instructions such as:

- `vmlava.s8`
- `vldrb.*`

Those are the clearest sign that the binary still contains the intended MVE dot-product path.

## Cycle Profiling

For real M55 bring-up, the repo can optionally instrument the main phases with the DWT cycle counter.

Enable it in the Cortex-M55 build with:

```sh
cmake -S . -B build-m55-prof \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-none-eabi-gcc.cmake \
  -DARM_M55_QEMU=ON \
  -DARM_PROFILE_CYCLES=ON
cmake --build build-m55-prof -j4
```

When `ARM_PROFILE_CYCLES=ON` is enabled on an M55 build:

- `g_cycle_profile` in `src/main.c` is populated so a debugger can inspect it after inference
- the program prints a short summary after the checksum
- the stats are broken down into:
  - full model cycles
  - input fill
  - classifier
  - per-layer total and FFN cycles
  - per-head KV-cache, Q-projection/RoPE, score-tile, softmax/value-fold, normalization, and output-projection cycles

This is intended for real hardware measurement. QEMU may compile and run the profiling code, but its counters are not meaningful for M55 performance.
