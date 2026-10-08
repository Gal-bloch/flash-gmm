"""
Flash-GMM E-step — H100-tuned two-launch mixed-precision kernel.

Two Triton launches (same structure as baseline):
  Pass 1: tiled online log-sum-exp over K -> writes logZ to HBM (in log2 space)
  Pass 2: responsibilities + atomic-add into Nk / mu_acc / sig_acc

H100 wins relative to baseline:
  - Adaptive tensor-core precision: TF32 for small batches and a centered
    BF16 metric for well-populated large batches, always with FP32 accumulation
  - Shape-specialized launch configurations for short and long row dimensions
  - Per-K constants precomputed on host (mu_sq, inv_s, log_pi offsets) and
    folded into log2 space:
      ll2[n,k] = c0[k] - ish[k] * x_sq[n] + (2*ish[k]) * xmu[n,k]      (log2 units)
  - tl.exp / tl.log replaced by tl.exp2 / tl.log2 -> H100 SFU ex2.approx fast paths.
  - X stays in its caller-owned FP32 allocation and is converted tile-by-tile.
    This avoids the O(ND) copy used by the earlier H100 variant.

Entry point (DO NOT RENAME):
    flash_gmm_estep(X, mu, log_sigma_sq, log_pi) -> (logZ, Nk, mu_acc, sig_acc)

sig_acc is the scatter around the mean supplied to the E-step,
``sum_i r_ik ||x_i - mu_in_k||^2``. The M-step applies the exact mean-shift
identity to center it around the updated mean. Statistics from streamed chunks
remain additive as long as every chunk uses the same ``mu_in``.
"""

from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl


SIG_ACC_KIND = "old_mean_scatter"
BF16_MIN_ROWS_PER_COMPONENT = 768


# ---------------------------------------------------------------------------
# Pass 1: online log-sum-exp over K (writes logZ to HBM, in log2 units)
# ---------------------------------------------------------------------------

@triton.jit
def _gmm_lse_kernel(
    X_ptr, mu_ptr, center_ptr, c0_ptr, ish_ptr,
    logZ2_ptr,
    N, K, D,
    stride_xn, stride_xd,
    stride_mk, stride_md,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    K_EXACT: tl.constexpr,
    USE_BF16_METRIC: tl.constexpr,
):
    pid = tl.program_id(0)
    n_off = (pid * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    n_mask = n_off < N
    d_off = tl.arange(0, BLOCK_D)

    # Large batches use a self-consistent centered BF16 metric. Small batches
    # retain TF32 because their tiny per-component populations amplify BF16
    # perturbations during the M-step.
    x = tl.load(
        X_ptr + n_off[:, None] * stride_xn + d_off[None, :] * stride_xd,
        mask=n_mask[:, None], other=0.0,
    )
    center = tl.load(center_ptr + d_off)
    x_centered = x - center[None, :]
    if USE_BF16_METRIC:
        x_operand = x_centered.to(tl.bfloat16)
        x_metric = x_operand.to(tl.float32)
    else:
        x_operand = x_centered
        x_metric = x_centered
    x_sq = tl.sum(x_metric * x_metric, axis=1)

    running_max = tl.full([BLOCK_N], float('-inf'), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_off = k0 + tl.arange(0, BLOCK_K)

        # K_EXACT: K divides BLOCK_K -> drop k_mask (production case K=1024).
        if K_EXACT:
            mu = tl.load(mu_ptr + k_off[:, None] * stride_mk + d_off[None, :] * stride_md)
            c0 = tl.load(c0_ptr + k_off)
            ish = tl.load(ish_ptr + k_off)
        else:
            k_mask = k_off < K
            mu = tl.load(
                mu_ptr + k_off[:, None] * stride_mk + d_off[None, :] * stride_md,
                mask=k_mask[:, None], other=0.0,
            )
            c0 = tl.load(c0_ptr + k_off, mask=k_mask, other=float('-inf'))
            ish = tl.load(ish_ptr + k_off, mask=k_mask, other=0.0)

        if USE_BF16_METRIC:
            xmu = tl.dot(x_operand, tl.trans(mu), out_dtype=tl.float32)
        else:
            xmu = tl.dot(
                x_operand, tl.trans(mu), input_precision="tf32",
                out_dtype=tl.float32,
            )

        ll2 = (c0[None, :] - ish[None, :] * x_sq[:, None]
               + (2.0 * ish[None, :]) * xmu)
        if not K_EXACT:
            ll2 = tl.where((k_off < K)[None, :], ll2, float('-inf'))

        block_max = tl.max(ll2, axis=1)
        new_max = tl.maximum(running_max, block_max)
        running_sum = (running_sum * tl.exp2(running_max - new_max)
                       + tl.sum(tl.exp2(ll2 - new_max[:, None]), axis=1))
        running_max = new_max

    log_Z2 = running_max + tl.log2(running_sum)
    tl.store(logZ2_ptr + n_off, log_Z2, mask=n_mask)


# ---------------------------------------------------------------------------
# Pass 2: responsibilities + atomic accumulation
# ---------------------------------------------------------------------------

@triton.jit
def _gmm_atomic_accum(
    X_ptr, mu_ptr, center_ptr, c0_ptr, ish_ptr, mu_sq_ptr, logZ2_ptr,
    Nk_ptr, mu_acc_ptr, sig_acc_ptr,
    N, K, D,
    stride_xn, stride_xd,
    stride_mk, stride_md,
    stride_ma_k, stride_ma_d,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    K_EXACT: tl.constexpr,
    USE_BF16_METRIC: tl.constexpr,
):
    pid = tl.program_id(0)
    n_off = (pid * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    n_mask = n_off < N
    d_off = tl.arange(0, BLOCK_D)

    # Responsibilities use the same BF16-rounded metric as pass 1.
    x = tl.load(
        X_ptr + n_off[:, None] * stride_xn + d_off[None, :] * stride_xd,
        mask=n_mask[:, None], other=0.0,
    )
    log_Z2 = tl.load(logZ2_ptr + n_off, mask=n_mask, other=0.0)
    center = tl.load(center_ptr + d_off)
    x_centered = x - center[None, :]
    if USE_BF16_METRIC:
        x_operand = x_centered.to(tl.bfloat16)
        x_metric = x_operand.to(tl.float32)
    else:
        x_operand = x_centered
        x_metric = x_centered
    x_sq = tl.sum(x_metric * x_metric, axis=1)

    for k0 in range(0, K, BLOCK_K):
        k_off = k0 + tl.arange(0, BLOCK_K)

        if K_EXACT:
            mu = tl.load(mu_ptr + k_off[:, None] * stride_mk + d_off[None, :] * stride_md)
            c0 = tl.load(c0_ptr + k_off)
            ish = tl.load(ish_ptr + k_off)
            mu_sq = tl.load(mu_sq_ptr + k_off)
        else:
            k_mask = k_off < K
            mu = tl.load(
                mu_ptr + k_off[:, None] * stride_mk + d_off[None, :] * stride_md,
                mask=k_mask[:, None], other=0.0,
            )
            c0 = tl.load(c0_ptr + k_off, mask=k_mask, other=float('-inf'))
            ish = tl.load(ish_ptr + k_off, mask=k_mask, other=0.0)
            mu_sq = tl.load(mu_sq_ptr + k_off, mask=k_mask, other=0.0)

        if USE_BF16_METRIC:
            xmu = tl.dot(x_operand, tl.trans(mu), out_dtype=tl.float32)
        else:
            xmu = tl.dot(
                x_operand, tl.trans(mu), input_precision="tf32",
                out_dtype=tl.float32,
            )
        sq = x_sq[:, None] - 2.0 * xmu + mu_sq[None, :]

        ll2 = (c0[None, :] - ish[None, :] * x_sq[:, None]
               + (2.0 * ish[None, :]) * xmu)
        if K_EXACT:
            ll2 = tl.where(n_mask[:, None], ll2, float('-inf'))
        else:
            ll2 = tl.where((k_off < K)[None, :] & n_mask[:, None], ll2, float('-inf'))

        r = tl.exp2(ll2 - log_Z2[:, None])

        nk_block = tl.sum(r, axis=0)
        # Centering before accumulation avoids catastrophic cancellation in
        # tight clusters. The M-step corrects from mu_in to the updated mean.
        sig_block = tl.sum(r * sq, axis=0)
        if K_EXACT:
            tl.atomic_add(Nk_ptr + k_off, nk_block)
            tl.atomic_add(sig_acc_ptr + k_off, sig_block)
        else:
            k_mask = k_off < K
            tl.atomic_add(Nk_ptr + k_off, nk_block, mask=k_mask)
            tl.atomic_add(sig_acc_ptr + k_off, sig_block, mask=k_mask)

        if USE_BF16_METRIC:
            r_operand = r.to(tl.bfloat16)
            mu_block = tl.dot(
                tl.trans(r_operand), x_operand, out_dtype=tl.float32
            )
        else:
            mu_block = tl.dot(
                tl.trans(r), x_operand, input_precision="tf32",
                out_dtype=tl.float32,
            )
        # mu_acc allocated as [K, BLOCK_D] -> drop d_mask, full-width store.
        if K_EXACT:
            tl.atomic_add(
                mu_acc_ptr + k_off[:, None] * stride_ma_k + d_off[None, :] * stride_ma_d,
                mu_block,
            )
        else:
            tl.atomic_add(
                mu_acc_ptr + k_off[:, None] * stride_ma_k + d_off[None, :] * stride_ma_d,
                mu_block,
                mask=(k_off < K)[:, None],
            )


# ---------------------------------------------------------------------------
# Public entry point — DO NOT RENAME
# ---------------------------------------------------------------------------

_LOG_2PI = 1.8378770664093453
_LOG2_E = 1.4426950408889634   # 1 / ln(2)


def flash_gmm_estep(
    X: torch.Tensor,
    mu: torch.Tensor,
    log_sigma_sq: torch.Tensor,
    log_pi: torch.Tensor,
):
    """Compute the GMM E-step sufficient statistics on GPU.

    Returns (logZ, Nk, mu_acc, sig_acc).
    """
    N, D = X.shape
    K = mu.shape[0]
    assert X.is_cuda and mu.is_cuda

    BLOCK_D = triton.next_power_of_2(D) if D > 0 else 1
    if BLOCK_D < 16:
        BLOCK_D = 16

    # Pad only when D requires it. In the paper workload D=128, X is passed
    # through without an O(ND) copy.
    # BF16 distance perturbations are amplified by the near-zero variances that
    # occur when the initialization has only a few samples per component.  Gate
    # on population density rather than total N so arbitrary K shapes remain
    # accurate while the large, well-populated paper regimes use tensor cores.
    use_bf16_metric = N >= K * BF16_MIN_ROWS_PER_COMPONENT
    center = (
        mu.mean(dim=0)
        if use_bf16_metric else torch.zeros(D, device=mu.device, dtype=mu.dtype)
    )
    if D == BLOCK_D:
        X_pad = X.contiguous()
        center_pad = center.contiguous()
        mu_centered = mu - center
        mu_pad = (
            mu_centered.to(torch.bfloat16).contiguous()
            if use_bf16_metric else mu_centered.contiguous()
        )
    else:
        X_pad = torch.zeros(N, BLOCK_D, device=X.device, dtype=X.dtype)
        X_pad[:, :D] = X
        center_pad = torch.zeros(BLOCK_D, device=mu.device, dtype=mu.dtype)
        center_pad[:D] = center
        mu_pad = torch.zeros(
            K, BLOCK_D, device=mu.device,
            dtype=torch.bfloat16 if use_bf16_metric else mu.dtype,
        )
        mu_pad[:, :D] = (mu - center).to(mu_pad.dtype)

    # Per-K constants in log2 space:
    inv_s = torch.exp(-log_sigma_sq)
    mu_metric = mu_pad.float()
    mu_sq = (mu_metric * mu_metric).sum(dim=1)                           # [K]
    ish = (0.5 * inv_s * _LOG2_E).contiguous()                         # [K]
    c0 = ((log_pi - 0.5 * D * (_LOG_2PI + log_sigma_sq)) * _LOG2_E
          - ish * mu_sq).contiguous()                                  # [K]
    mu_sq = mu_sq.contiguous()

    logZ2 = torch.empty(N, device=X.device, dtype=torch.float32)
    Nk = torch.zeros(K, device=X.device, dtype=torch.float32)
    # mu_acc allocated padded -> output is contiguous slice.
    mu_acc_pad = torch.zeros(K, BLOCK_D, device=X.device, dtype=torch.float32)
    sig_acc = torch.zeros(K, device=X.device, dtype=torch.float32)

    # These two regimes are the stable winners on H100 for K=1024,D=128.
    # Fixed launch shapes also avoid compiling dozens of invalid/slow variants
    # independently for every N in a scale sweep.
    if not use_bf16_metric:
        lse_launch = dict(BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=2)
        accum_launch = dict(BLOCK_N=128, BLOCK_K=64, num_warps=8, num_stages=2)
    else:
        # Environment overrides are intentionally undocumented tuning hooks;
        # defaults below are the validated production configuration.
        lse_launch = dict(
            BLOCK_N=int(os.environ.get("FLASH_GMM_LSE_BN", 128)),
            BLOCK_K=int(os.environ.get("FLASH_GMM_LSE_BK", 64)),
            num_warps=int(os.environ.get("FLASH_GMM_LSE_WARPS", 4)),
            num_stages=int(os.environ.get("FLASH_GMM_LSE_STAGES", 2)),
        )
        accum_launch = dict(
            BLOCK_N=int(os.environ.get("FLASH_GMM_ACCUM_BN", 512)),
            BLOCK_K=int(os.environ.get("FLASH_GMM_ACCUM_BK", 32)),
            num_warps=int(os.environ.get("FLASH_GMM_ACCUM_WARPS", 8)),
            num_stages=int(os.environ.get("FLASH_GMM_ACCUM_STAGES", 1)),
        )

    # The unmasked path is valid for the production launch shapes where the
    # larger K tile is a multiple of the smaller one.
    K_EXACT = (K % max(lse_launch["BLOCK_K"], accum_launch["BLOCK_K"]) == 0)

    _gmm_lse_kernel[(triton.cdiv(N, lse_launch["BLOCK_N"]),)](
        X_pad, mu_pad, center_pad, c0, ish, logZ2,
        N, K, D,
        X_pad.stride(0), X_pad.stride(1),
        mu_pad.stride(0), mu_pad.stride(1),
        BLOCK_D=BLOCK_D, K_EXACT=K_EXACT,
        USE_BF16_METRIC=use_bf16_metric, **lse_launch,
    )

    _gmm_atomic_accum[(triton.cdiv(N, accum_launch["BLOCK_N"]),)](
        X_pad, mu_pad, center_pad, c0, ish, mu_sq, logZ2,
        Nk, mu_acc_pad, sig_acc,
        N, K, D,
        X_pad.stride(0), X_pad.stride(1),
        mu_pad.stride(0), mu_pad.stride(1),
        mu_acc_pad.stride(0), mu_acc_pad.stride(1),
        BLOCK_D=BLOCK_D, K_EXACT=K_EXACT,
        USE_BF16_METRIC=use_bf16_metric, **accum_launch,
    )

    # Return additive sufficient statistics, including old-mean scatter.
    mu_acc = (mu_acc_pad[:, :D] + Nk[:, None] * center[None, :]).contiguous()
    # Convert in place: allocating a second N-vector would double the only
    # O(N) workspace at 50M/100M rows.
    logZ = logZ2.mul_(1.0 / _LOG2_E)
    return logZ, Nk, mu_acc, sig_acc


def flash_gmm_mstep(Nk, mu_acc, sig_acc, mu_in, D, eps=1e-10, var_floor=1e-6):
    """
    M-step from the E-step's sufficient statistics.

    sig_acc is the scatter about ``mu_in``. The parallel-axis identity gives
    the scatter about the updated mean without subtracting two large raw
    moments, which is substantially more stable in fp32.

    Returns (mu, log_sigma_sq, log_pi).
    """
    Nk_safe = Nk.clamp(min=eps)
    log_pi  = torch.log((Nk / Nk.sum()).clamp_min(eps))
    mu_new  = mu_acc / Nk_safe[:, None]
    centered = sig_acc - Nk * ((mu_new - mu_in) ** 2).sum(dim=1)
    var = (centered / (D * Nk_safe)).clamp(min=var_floor)

    return mu_new, torch.log(var), log_pi
