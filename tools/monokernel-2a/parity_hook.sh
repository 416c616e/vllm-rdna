#!/usr/bin/env bash
# LOAD_HOOK: greedy top-5 logprob dump for the off/on parity check, plus the
# mono path decisions from the serve log. Exported: URL PORT SERVED MODEL OUT.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
"${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.31.0-final}/bin/python" "$HERE/logprob_parity.py" \
    dump --url "http://127.0.0.1:$PORT" --model "$SERVED" --out "$OUT/parity.json" || { echo FAIL; exit 1; }
grep -a "rdna_mono_" "$OUT/serve.log" | sed 's/^.*rdna_mono_/rdna_mono_/' | sort | uniq -c | head -40
echo PASS
