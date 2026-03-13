#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "arm_math.h"
#include "model_data.h"

static q7_t g_hidden_a[INPUT_SEQ_LEN * MODEL_DIM];
static q7_t g_hidden_b[INPUT_SEQ_LEN * MODEL_DIM];
static q7_t g_output[OUTPUT_SEQ_LEN * NUM_CLASSES];

static q7_t g_head_k_cache[INPUT_SEQ_LEN * HEAD_DIM];
static q7_t g_head_v_cache[INPUT_SEQ_LEN * HEAD_DIM];
static q7_t g_query_tile[QUERY_TILE * HEAD_DIM];
static q7_t g_head_context_tile[QUERY_TILE * HEAD_DIM];
static q7_t g_context_tile[QUERY_TILE * MODEL_DIM];
static q7_t g_work_tile[QUERY_TILE * MODEL_DIM];
static q7_t g_residual_tile[QUERY_TILE * MODEL_DIM];
static q7_t g_classifier_tile[QUERY_TILE * NUM_CLASSES];
static q7_t g_matmul_state[MODEL_DIM * ((MODEL_DIM > HEAD_DIM) ? MODEL_DIM : HEAD_DIM)];
static int16_t g_row_max[QUERY_TILE];
static int32_t g_row_sum[QUERY_TILE];
static int32_t g_row_acc[QUERY_TILE * HEAD_DIM];

static q7_t sat_q7(int32_t value)
{
    if (value > 127) {
        return 127;
    }
    if (value < -128) {
        return -128;
    }
    return (q7_t)value;
}

static uint32_t min_u32(uint32_t a, uint32_t b)
{
    return (a < b) ? a : b;
}

static uint32_t xorshift32(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static uint16_t softmax_decay_q12(int32_t delta)
{
    if (delta <= 0) {
        return SOFTMAX_Q12_ONE;
    }
    if (delta >= SOFTMAX_LUT_SIZE) {
        return 0;
    }
    return g_softmax_exp_q12[delta];
}

static void check_status(arm_status status, const char *what)
{
    if (status != ARM_MATH_SUCCESS) {
        fprintf(stderr, "%s failed with status %d\n", what, (int)status);
        exit(1);
    }
}

static void q7_linear_block(
    const q7_t *input_block,
    uint16_t rows,
    uint16_t cols_in,
    const q7_t *weights,
    uint16_t cols_out,
    const q7_t *bias,
    q7_t *output_block)
{
    arm_matrix_instance_q7 a = { rows, cols_in, (q7_t *)input_block };
    arm_matrix_instance_q7 b = { cols_in, cols_out, (q7_t *)weights };
    arm_matrix_instance_q7 c = { rows, cols_out, output_block };

    /* Project one contiguous tile with CMSIS-DSP's q7 matrix multiply. */
    check_status(arm_mat_mult_q7(&a, &b, &c, g_matmul_state), "arm_mat_mult_q7");

    if (bias == NULL) {
        return;
    }

    /* Add the per-channel bias row by row with the q7 vector add primitive. */
    for (uint32_t row = 0; row < rows; ++row) {
        arm_add_q7(
            output_block + row * cols_out,
            bias,
            output_block + row * cols_out,
            cols_out);
    }
}

static void add_bias_rows(q7_t *block, uint16_t rows, uint16_t cols, const q7_t *bias)
{
    for (uint32_t row = 0; row < rows; ++row) {
        arm_add_q7(
            block + row * cols,
            bias,
            block + row * cols,
            cols);
    }
}

static int16_t dot_q7_head(const q7_t *a, const q7_t *b)
{
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE) && (HEAD_DIM == 16)
    q7x16_t vec_a = vld1q(a);
    q7x16_t vec_b = vld1q(b);
    q31_t score_acc = vmladavaq(0, vec_a, vec_b);
#else
    q31_t score_acc = 0;
    arm_dot_prod_q7(a, b, HEAD_DIM, &score_acc);
#endif
    return (int16_t)(score_acc >> ATTN_SCORE_SHIFT);
}

static void init_row_acc_from_value_q12(int32_t *acc, const q7_t *value_vec)
{
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE) && (HEAD_DIM == 16)
    q31x4_t val0 = vldrbq_s32(value_vec + 0);
    q31x4_t val1 = vldrbq_s32(value_vec + 4);
    q31x4_t val2 = vldrbq_s32(value_vec + 8);
    q31x4_t val3 = vldrbq_s32(value_vec + 12);

    vstrwq_s32(acc + 0, vmulq_n_s32(val0, SOFTMAX_Q12_ONE));
    vstrwq_s32(acc + 4, vmulq_n_s32(val1, SOFTMAX_Q12_ONE));
    vstrwq_s32(acc + 8, vmulq_n_s32(val2, SOFTMAX_Q12_ONE));
    vstrwq_s32(acc + 12, vmulq_n_s32(val3, SOFTMAX_Q12_ONE));
#else
    for (uint32_t d = 0; d < HEAD_DIM; ++d) {
        acc[d] = SOFTMAX_Q12_ONE * value_vec[d];
    }
#endif
}

static void scale_row_acc_q12_inplace(int32_t *acc, uint16_t alpha_q12)
{
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE) && (HEAD_DIM == 16)
    q31x4_t alpha = vdupq_n_s32((int32_t)alpha_q12);
    q31x4_t round = vdupq_n_s32(SOFTMAX_Q12_ONE / 2);
    q31x4_t acc0 = vld1q(acc + 0);
    q31x4_t acc1 = vld1q(acc + 4);
    q31x4_t acc2 = vld1q(acc + 8);
    q31x4_t acc3 = vld1q(acc + 12);

    acc0 = vshrq(vaddq(vmulq(acc0, alpha), round), 12);
    acc1 = vshrq(vaddq(vmulq(acc1, alpha), round), 12);
    acc2 = vshrq(vaddq(vmulq(acc2, alpha), round), 12);
    acc3 = vshrq(vaddq(vmulq(acc3, alpha), round), 12);

    vstrwq_s32(acc + 0, acc0);
    vstrwq_s32(acc + 4, acc1);
    vstrwq_s32(acc + 8, acc2);
    vstrwq_s32(acc + 12, acc3);
#else
    for (uint32_t d = 0; d < HEAD_DIM; ++d) {
        acc[d] = (acc[d] * alpha_q12 + (SOFTMAX_Q12_ONE / 2)) >> 12;
    }
#endif
}

static void add_weighted_value_q12(int32_t *acc, const q7_t *value_vec, uint16_t weight_q12)
{
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE) && (HEAD_DIM == 16)
    q31x4_t val0 = vldrbq_s32(value_vec + 0);
    q31x4_t val1 = vldrbq_s32(value_vec + 4);
    q31x4_t val2 = vldrbq_s32(value_vec + 8);
    q31x4_t val3 = vldrbq_s32(value_vec + 12);
    q31x4_t acc0 = vld1q(acc + 0);
    q31x4_t acc1 = vld1q(acc + 4);
    q31x4_t acc2 = vld1q(acc + 8);
    q31x4_t acc3 = vld1q(acc + 12);

    acc0 = vaddq(acc0, vmulq_n_s32(val0, (int32_t)weight_q12));
    acc1 = vaddq(acc1, vmulq_n_s32(val1, (int32_t)weight_q12));
    acc2 = vaddq(acc2, vmulq_n_s32(val2, (int32_t)weight_q12));
    acc3 = vaddq(acc3, vmulq_n_s32(val3, (int32_t)weight_q12));

    vstrwq_s32(acc + 0, acc0);
    vstrwq_s32(acc + 4, acc1);
    vstrwq_s32(acc + 8, acc2);
    vstrwq_s32(acc + 12, acc3);
#else
    for (uint32_t d = 0; d < HEAD_DIM; ++d) {
        acc[d] += weight_q12 * value_vec[d];
    }
#endif
}

static void relu_q7_inplace(q7_t *data, uint32_t count)
{
    /* ReLU is just q7 clipping to the non-negative range. */
    arm_clip_q7(data, data, 0, 127, count);
}

static void normalize_row_acc_to_q7(const int32_t *row_acc, int32_t row_sum, q7_t *dst)
{
    if (row_sum == 0) {
        arm_fill_q7(0, dst, HEAD_DIM);
        return;
    }

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE) && (HEAD_DIM == 16)
    q31x4_t half = vdupq_n_s32(row_sum / 2);

    for (uint32_t chunk = 0; chunk < HEAD_DIM; chunk += 4) {
        q31x4_t numer = vld1q(row_acc + chunk);
        q31x4_t pos = vaddq(numer, half);
        q31x4_t neg = vsubq(numer, half);
        q31x4_t adjusted = vpselq(pos, neg, vcmpgeq_n_s32(numer, 0));

        for (uint32_t lane = 0; lane < 4; ++lane) {
            dst[chunk + lane] = sat_q7(vgetq_lane_s32(adjusted, lane) / row_sum);
        }
    }
#else
    for (uint32_t d = 0; d < HEAD_DIM; ++d) {
        int32_t numerator = row_acc[d];
        int32_t rounded = (numerator >= 0)
            ? (numerator + row_sum / 2) / row_sum
            : (numerator - row_sum / 2) / row_sum;

        dst[d] = sat_q7(rounded);
    }
#endif
}

static void apply_rope_q7_inplace(q7_t *data, uint32_t rows, uint32_t start_pos)
{
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE) && (HEAD_DIM == 16)
    static const uint32_t k_even_offsets[4] = { 0U, 2U, 4U, 6U };
    static const uint32_t k_odd_offsets[4] = { 1U, 3U, 5U, 7U };
    uint32x4_t even_offsets = vld1q(k_even_offsets);
    uint32x4_t odd_offsets = vld1q(k_odd_offsets);
    q31x4_t round_q15 = vdupq_n_s32(1 << (ROPE_Q15_SHIFT - 1));
#endif

    for (uint32_t row = 0; row < rows; ++row) {
        q7_t *row_ptr = data + row * HEAD_DIM;
        uint32_t pos = start_pos + row;

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE) && (HEAD_DIM == 16)
        const q15_t *cos_ptr = &g_rope_cos_q15[pos][0];
        const q15_t *sin_ptr = &g_rope_sin_q15[pos][0];

        for (uint32_t pair = 0; pair < ROPE_PAIR_DIM; pair += 4) {
            const int8_t *pair_ptr = (const int8_t *)(row_ptr + 2U * pair);
            q31x4_t x0 = vldrbq_gather_offset_s32(pair_ptr, even_offsets);
            q31x4_t x1 = vldrbq_gather_offset_s32(pair_ptr, odd_offsets);
            q31x4_t cos_q15 = vldrhq_s32(cos_ptr + pair);
            q31x4_t sin_q15 = vldrhq_s32(sin_ptr + pair);
            q31x4_t y0 = vsubq(vmulq(x0, cos_q15), vmulq(x1, sin_q15));
            q31x4_t y1 = vaddq(vmulq(x0, sin_q15), vmulq(x1, cos_q15));

            y0 = vshrq(vaddq(y0, round_q15), ROPE_Q15_SHIFT);
            y1 = vshrq(vaddq(y1, round_q15), ROPE_Q15_SHIFT);

            for (uint32_t lane = 0; lane < 4; ++lane) {
                row_ptr[2U * (pair + lane)] = sat_q7(vgetq_lane_s32(y0, lane));
                row_ptr[2U * (pair + lane) + 1U] = sat_q7(vgetq_lane_s32(y1, lane));
            }
        }
#else
        for (uint32_t pair = 0; pair < ROPE_PAIR_DIM; ++pair) {
            int32_t x0 = row_ptr[2U * pair];
            int32_t x1 = row_ptr[2U * pair + 1U];
            int32_t cos_q15 = g_rope_cos_q15[pos][pair];
            int32_t sin_q15 = g_rope_sin_q15[pos][pair];
            int32_t y0 = (x0 * cos_q15 - x1 * sin_q15 + (1 << (ROPE_Q15_SHIFT - 1))) >> ROPE_Q15_SHIFT;
            int32_t y1 = (x0 * sin_q15 + x1 * cos_q15 + (1 << (ROPE_Q15_SHIFT - 1))) >> ROPE_Q15_SHIFT;

            row_ptr[2U * pair] = sat_q7(y0);
            row_ptr[2U * pair + 1U] = sat_q7(y1);
        }
#endif
    }
}

static void fill_demo_input(q7_t *sequence)
{
    uint32_t state = 0x1A2B3C4Du;

    /* Build a deterministic synthetic sequence so the program is self-contained. */
    for (uint32_t token = 0; token < INPUT_SEQ_LEN; ++token) {
        for (uint32_t feature = 0; feature < MODEL_DIM; ++feature) {
            int32_t trend = (int32_t)((token * 3 + feature * 11) % 29) - 14;
            int32_t stripe = ((token + feature) & 1U) ? 7 : -7;
            int32_t noise = (int32_t)(xorshift32(&state) & 0x0F) - 8;
            sequence[token * MODEL_DIM + feature] = sat_q7(trend + stripe + noise);
        }
    }
}

static void run_attention_head_block_with_cache(
    const q7_t *input_sequence,
    const model_layer_t *layer,
    uint32_t head_idx,
    uint32_t query_start,
    uint32_t query_rows,
    q7_t *head_context_tile)
{
    /* Project just the active query tile for this head. */
    q7_linear_block(
        input_sequence + query_start * MODEL_DIM,
        (uint16_t)query_rows,
        MODEL_DIM,
        &layer->w_q[head_idx][0][0],
        HEAD_DIM,
        NULL,
        g_query_tile);

    /* Apply RoPE to the active query tile using absolute token positions. */
    apply_rope_q7_inplace(g_query_tile, query_rows, query_start);

    /* Seed each query row from the first key so the steady-state loop has no empty-row branch. */
    for (uint32_t q = 0; q < query_rows; ++q) {
        const q7_t *query_vec = &g_query_tile[q * HEAD_DIM];
        const q7_t *first_key_vec = &g_head_k_cache[0];
        const q7_t *first_value_vec = &g_head_v_cache[0];

        g_row_max[q] = dot_q7_head(query_vec, first_key_vec);
        g_row_sum[q] = SOFTMAX_Q12_ONE;
        init_row_acc_from_value_q12(&g_row_acc[q * HEAD_DIM], first_value_vec);
    }

    for (uint32_t key_start = 0; key_start < INPUT_SEQ_LEN; key_start += KEY_TILE) {
        uint32_t key_offset = (key_start == 0U) ? 1U : 0U;
        uint32_t key_rows = min_u32(KEY_TILE, INPUT_SEQ_LEN - key_start);
        const q7_t *key_tile = &g_head_k_cache[key_start * HEAD_DIM];
        const q7_t *value_tile = &g_head_v_cache[key_start * HEAD_DIM];

        for (uint32_t q = 0; q < query_rows; ++q) {
            const q7_t *query_vec = &g_query_tile[q * HEAD_DIM];
            int16_t row_max = g_row_max[q];
            int32_t row_sum = g_row_sum[q];
            int32_t *row_acc = &g_row_acc[q * HEAD_DIM];
            const q7_t *key_vec = key_tile + key_offset * HEAD_DIM;
            const q7_t *value_vec = value_tile + key_offset * HEAD_DIM;

            for (uint32_t k = key_offset; k < key_rows; ++k) {
                int16_t score = dot_q7_head(query_vec, key_vec);
                uint16_t weight_q12;

                /* Update the online softmax max and renormalize old partial sums if the max grows. */
                if (score > row_max) {
                    uint16_t alpha_q12 = softmax_decay_q12((int32_t)score - row_max);
                    row_sum = (row_sum * alpha_q12 + (SOFTMAX_Q12_ONE / 2)) >> 12;
                    scale_row_acc_q12_inplace(row_acc, alpha_q12);
                    row_max = score;
                    weight_q12 = SOFTMAX_Q12_ONE;
                } else {
                    weight_q12 = softmax_decay_q12((int32_t)row_max - score);
                }

                /* Fold the new V contribution into the unnormalized weighted output. */
                row_sum += weight_q12;
                add_weighted_value_q12(row_acc, value_vec, weight_q12);
                key_vec += HEAD_DIM;
                value_vec += HEAD_DIM;
            }

            g_row_max[q] = row_max;
            g_row_sum[q] = row_sum;
        }
    }

    for (uint32_t q = 0; q < query_rows; ++q) {
        /* Store this head's normalized context into a compact 16-wide tile scratch. */
        normalize_row_acc_to_q7(
            &g_row_acc[q * HEAD_DIM],
            g_row_sum[q],
            &head_context_tile[q * HEAD_DIM]);
    }
}

static void build_head_kv_cache(const q7_t *input_sequence, const model_layer_t *layer, uint32_t head_idx)
{
    /* Materialize this head's full K sequence once so all query tiles can reuse it. */
    q7_linear_block(
        input_sequence,
        INPUT_SEQ_LEN,
        MODEL_DIM,
        &layer->w_k[head_idx][0][0],
        HEAD_DIM,
        NULL,
        g_head_k_cache);

    /* Apply RoPE once to the cached K rows so all query tiles reuse the rotated keys. */
    apply_rope_q7_inplace(g_head_k_cache, INPUT_SEQ_LEN, 0U);

    /* Materialize this head's full V sequence once for the same reason. */
    q7_linear_block(
        input_sequence,
        INPUT_SEQ_LEN,
        MODEL_DIM,
        &layer->w_v[head_idx][0][0],
        HEAD_DIM,
        NULL,
        g_head_v_cache);
}

static void run_transformer_layer(const q7_t *current_sequence, q7_t *next_sequence, const model_layer_t *layer)
{
    /* Accumulate the output-projected attention result directly into the next 24-wide sequence buffer. */
    arm_fill_q7(0, next_sequence, INPUT_SEQ_LEN * MODEL_DIM);

    for (uint32_t head_idx = 0; head_idx < NUM_HEADS; ++head_idx) {
        build_head_kv_cache(current_sequence, layer, head_idx);

        for (uint32_t query_start = 0; query_start < INPUT_SEQ_LEN; query_start += QUERY_TILE) {
            uint32_t query_rows = min_u32(QUERY_TILE, INPUT_SEQ_LEN - query_start);
            uint32_t tile_elems = query_rows * MODEL_DIM;

            /* Run this head against all cached K/V rows and emit one 16-wide context tile. */
            run_attention_head_block_with_cache(
                current_sequence,
                layer,
                head_idx,
                query_start,
                query_rows,
                g_head_context_tile);

            /* Project this head through its W_o slice and accumulate into the 24-wide output tile. */
            q7_linear_block(
                g_head_context_tile,
                (uint16_t)query_rows,
                HEAD_DIM,
                &layer->w_o[head_idx * HEAD_DIM][0],
                MODEL_DIM,
                NULL,
                g_context_tile);

            arm_add_q7(
                next_sequence + query_start * MODEL_DIM,
                g_context_tile,
                next_sequence + query_start * MODEL_DIM,
                tile_elems);
        }
    }

    for (uint32_t query_start = 0; query_start < INPUT_SEQ_LEN; query_start += QUERY_TILE) {
        uint32_t query_rows = min_u32(QUERY_TILE, INPUT_SEQ_LEN - query_start);
        uint32_t tile_elems = query_rows * MODEL_DIM;

        /* Apply the shared output bias once after all head slices have been accumulated. */
        add_bias_rows(next_sequence + query_start * MODEL_DIM, (uint16_t)query_rows, MODEL_DIM, layer->b_o);

        /* Add the attention residual directly on the current tile. */
        arm_add_q7(
            current_sequence + query_start * MODEL_DIM,
            next_sequence + query_start * MODEL_DIM,
            g_residual_tile,
            tile_elems);

        /* Apply the first 24->24 feed-forward projection on the residual tile. */
        q7_linear_block(
            g_residual_tile,
            (uint16_t)query_rows,
            MODEL_DIM,
            &layer->w_ff1[0][0],
            MODEL_DIM,
            layer->b_ff1,
            g_work_tile);

        /* Apply the non-linearity before the second 24->24 projection. */
        relu_q7_inplace(g_work_tile, tile_elems);

        /* Project back to the model width and reuse the tile scratch as FFN output. */
        q7_linear_block(
            g_work_tile,
            (uint16_t)query_rows,
            MODEL_DIM,
            &layer->w_ff2[0][0],
            MODEL_DIM,
            layer->b_ff2,
            g_context_tile);

        /* Overwrite the spare sequence buffer tile with the final layer output. */
        arm_add_q7(
            g_residual_tile,
            g_context_tile,
            next_sequence + query_start * MODEL_DIM,
            tile_elems);
    }
}

static void classify_sequence(const q7_t *sequence, q7_t *output)
{
    for (uint32_t output_start = 0; output_start < OUTPUT_SEQ_LEN; output_start += QUERY_TILE) {
        uint32_t output_rows = min_u32(QUERY_TILE, OUTPUT_SEQ_LEN - output_start);

        /* Apply the 24->4 output classifier directly on the final sequence tile. */
        q7_linear_block(
            sequence + output_start * MODEL_DIM,
            (uint16_t)output_rows,
            MODEL_DIM,
            &g_classifier_w[0][0],
            NUM_CLASSES,
            g_classifier_b,
            g_classifier_tile);

        arm_copy_q7(
            g_classifier_tile,
            output + output_start * NUM_CLASSES,
            output_rows * NUM_CLASSES);
    }
}

static void run_model(q7_t *buffer_a, q7_t *buffer_b, q7_t *output)
{
    q7_t *current = buffer_a;
    q7_t *next = buffer_b;

    /* Run the three stacked attention blocks in sequence, swapping the two sequence buffers. */
    for (uint32_t layer_idx = 0; layer_idx < NUM_LAYERS; ++layer_idx) {
        run_transformer_layer(current, next, &g_model_layers[layer_idx]);

        q7_t *tmp = current;
        current = next;
        next = tmp;
    }

    /* Emit one 24->4 classifier row per final sequence position. */
    classify_sequence(current, output);
}

int main(void)
{
    uint32_t checksum = 0;

    if ((NUM_HEADS * HEAD_DIM) != ATTN_DIM) {
        fprintf(stderr, "invalid head packing\n");
        return 1;
    }

    /* Populate the fixed-size input tensor. */
    fill_demo_input(g_hidden_a);

    /* Run the low-memory int8 model end to end. */
    run_model(g_hidden_a, g_hidden_b, g_output);

    /* Print a small prefix of the logits so the binary has a visible result. */
    for (uint32_t row = 0; row < min_u32(8U, OUTPUT_SEQ_LEN); ++row) {
        printf("%4lu:", (unsigned long)row);
        for (uint32_t col = 0; col < NUM_CLASSES; ++col) {
            q7_t value = g_output[row * NUM_CLASSES + col];
            checksum = checksum * 131u + (uint8_t)value;
            printf(" %4d", value);
        }
        printf("\n");
    }

    for (uint32_t row = 8; row < OUTPUT_SEQ_LEN; ++row) {
        for (uint32_t col = 0; col < NUM_CLASSES; ++col) {
            checksum = checksum * 131u + (uint8_t)g_output[row * NUM_CLASSES + col];
        }
    }

    printf("checksum: 0x%08lx\n", (unsigned long)checksum);
    return 0;
}
