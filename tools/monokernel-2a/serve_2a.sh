#!/usr/bin/env bash
# One flashnext-mtp2 serve (INT4 PLE) of this tree through serve_validate.sh,
# MonoKernel switch off or on.
#   ARM=on TAG=m2-on GPUS=2,3,4,5 PORT=18790 bash tools/monokernel-2a/serve_2a.sh [KEY=v ...]
# ARM=on adds --additional-config {"rdna_mono_ffn":true}. CELLS / BENCH_EXTRA /
# PPL_PROBE ... pass through to serve_validate.sh. The LOAD_HOOK dumps greedy
# top-5 logprobs (logprob_parity.py) for the off/on comparison.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(cd "$HERE/../.." && pwd)
RUN=${RUN:-$HOME/w4a8_runs/monokernel-2a}
: "${ARM:?ARM=on|off}" "${GPUS:?}"
export MODEL=${MODEL:-/home/chenco_adm/hfcache/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c}
PLE_QUANT_DIR=${PLE_QUANT_DIR:-/home/chenco_adm/hfcache/hub/models--primitive-ai--Qwen3.8-Flash-Next-PLE-quant/snapshots/4f861b63f69e61bfc2e22130ec91ec67f03ec43e/ples_int4}
export RECIPE=${RECIPE:-flashnext-mtp2} GPUS PORT=${PORT:-18790} TAG=${TAG:-2a-$ARM}
export OUT=$RUN/serve-$TAG CACHE=${CACHE:-$RUN/cache-2a-$ARM}
export VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.31.0-final}
export LOAD_HOOK=${LOAD_HOOK:-$HERE/parity_hook.sh}
if [[ $ARM == on ]]; then
    export EXTRA_ARGS="${EXTRA_ARGS:+$EXTRA_ARGS }--additional-config {\"rdna_mono_ffn\":true}"
fi
exec bash "$TREE/tools/rdna/port_v031/serve_validate.sh" \
    PLE=int4 PLE_QUANT_DIR="$PLE_QUANT_DIR" "$@"
