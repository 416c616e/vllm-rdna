# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbench + accuracy check: RDNA2 W4A8 vs W4A16 MoE GEMMs (one GPU).

Random int4 experts in the RDNA2 MoE layout ([E, K/8, N] shuffled-equivalent
random dwords, symmetric zeros), Flash-Next per-rank geometry by default
(EP4: 512 global / 128 local experts, top-10, hidden 2560, intermediate 640,
group 128). For each token count it times gate_up (w13) and down (w2, fused
top-k reduce) on both kernels, reports the first-call wall time (catches
per-shape one-off stalls) and the W4A8 relative error against W4A16.

  python tools/rdna/bench_moe_w4a8.py [--tokens 64,256,1023,2048,4096]
      [--w4a8-bm 8] [--json out.json]
"""

import argparse
import functools
import json
import time

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)


def make_experts(e, k, n, group, dev):
    w = torch.randint(
        -(2**31), 2**31 - 1, (e, k // 8, n), dtype=torch.int32, device=dev
    )
    scales = (torch.rand(e, k // group, n, device=dev) * 0.02 + 0.005).half()
    # GPTQv1 symmetric: stored nibble 7 -> zero point 8 with zero_offset 1.
    qzeros = torch.full(
        (e, k // group, n // 8), 0x77777777, dtype=torch.int32, device=dev
    )
    return w, scales, qzeros


def run(kind, bm, a, act, w13, w2, topk_ids, topk_w, local, emap, top_k):
    sorted_ids, expert_ids, ntp = moe_align_block_size(
        topk_ids, bm, local, emap, ignore_invalid_experts=True
    )
    m = a.size(0)
    c1 = torch.zeros(m * top_k, w13[0].size(2), dtype=torch.half, device=a.device)
    c2 = torch.zeros(m, w2[0].size(2), dtype=torch.half, device=a.device)
    empty = topk_w.new_empty(0)
    gemm = ops.moe_w4a8_gemm_rdna2 if kind == "w4a8" else ops.moe_gptq_gemm_rdna2
    gemm(
        a,
        c1,
        w13[0],
        w13[1],
        w13[2],
        empty,
        sorted_ids,
        expert_ids,
        ntp,
        top_k,
        bm,
        False,
        0,
    )
    gemm(
        act,
        c2,
        w2[0],
        w2[1],
        w2[2],
        topk_w,
        sorted_ids,
        expert_ids,
        ntp,
        1,
        bm,
        True,
        top_k,
    )
    return c1, c2


def timed(fn, iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="64,256,512,1023,1024,2048,4096")
    p.add_argument("--w4a8-bm", type=int, default=8)
    p.add_argument("--hidden", type=int, default=2560)
    p.add_argument("--inter", type=int, default=640)
    p.add_argument("--experts", type=int, default=512)
    p.add_argument("--local", type=int, default=128)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--group", type=int, default=128)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--json", default="")
    args = p.parse_args()

    dev = torch.device("cuda")
    torch.manual_seed(0)
    w13 = make_experts(args.local, args.hidden, 2 * args.inter, args.group, dev)
    w2 = make_experts(args.local, args.inter, args.hidden, args.group, dev)
    emap = torch.full((args.experts,), -1, dtype=torch.int32, device=dev)
    emap[: args.local] = torch.arange(args.local, dtype=torch.int32, device=dev)

    rows = []
    for m in [int(x) for x in args.tokens.split(",")]:
        a = torch.randn(m, args.hidden, dtype=torch.half, device=dev)
        act = torch.randn(m * args.top_k, args.inter, dtype=torch.half, device=dev)
        logits = torch.randn(m, args.experts, device=dev)
        topk_w, topk_ids = torch.topk(logits.softmax(-1), args.top_k, dim=-1)
        topk_ids = topk_ids.to(torch.int32)
        topk_w = topk_w.reshape(-1).float().contiguous()
        bm16 = 1 if m <= 4 else (8 if m >= 512 else 4)
        bm8 = args.w4a8_bm
        common = (a, act, w13, w2, topk_ids, topk_w, args.local, emap, args.top_k)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r8 = run("w4a8", bm8, *common)
        torch.cuda.synchronize()
        first_ms = (time.perf_counter() - t0) * 1e3
        r16 = run("w4a16", bm16, *common)
        torch.cuda.synchronize()

        def rel(x, y):
            return (
                (x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-9)
            ).item()

        err1, err2 = rel(r8[0], r16[0]), rel(r8[1], r16[1])
        t16 = timed(functools.partial(run, "w4a16", bm16, *common), args.iters)
        t8 = timed(functools.partial(run, "w4a8", bm8, *common), args.iters)
        row = dict(
            tokens=m,
            bm_w4a16=bm16,
            bm_w4a8=bm8,
            w4a16_ms=round(t16, 3),
            w4a8_ms=round(t8, 3),
            speedup=round(t16 / t8, 3),
            w4a8_first_call_ms=round(first_ms, 1),
            rel_err_w13=round(err1, 4),
            rel_err_w2=round(err2, 4),
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
