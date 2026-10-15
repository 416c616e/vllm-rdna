#!/usr/bin/env bash
# Stage driver: one serve per spec (recipe|arm|tag|extra config[|KEY=v ...]),
# in order,
# each after the GPUs are idle, with a clock/power log alongside and the
# final in-serve timing lines extracted to <serve dir>/timing.txt.
#   SPECS='mtp0|off-t|a0|"rdna_mono_ffn_max_rows":8;mtp0|on-t|a0|"rdna_mono_ffn_max_rows":8' \
#   CELLS="1024:512:1 1024:512:8" GPUS=6,7,8,9 PORT=18790 bash tools/monokernel-2a/run_specs.sh
# Serve dirs: ~/w4a8_runs/monokernel-2a/serve-<tag>-<recipe>-<arm>/.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RUN=${RUN:-$HOME/w4a8_runs/monokernel-2a}
: "${GPUS:?}" "${SPECS:?}"
CELLS=${CELLS:-"1024:512:1 1024:512:8"}
SMI_DEVS=${SMI_DEVS:-"0 1 2 3"}  # rocm-smi GPU[0-3] = HIP 6-9, GPU[4-7] = HIP 2-5
wait_idle() {
    for _ in $(seq 1 120); do
        busy=$(rocm-smi -d $SMI_DEVS --showmemuse 2>/dev/null | awk '/VRAM%/{s+=$NF} END{print s+0}')
        [[ $busy -eq 0 ]] && return 0
        sleep 10
    done
    return 1
}
IFS=';' read -ra list <<< "$SPECS"
for spec in "${list[@]}"; do
    IFS='|' read -r mtp arm tag cfg kv <<< "$spec"
    wait_idle || { echo "GPUs not idle"; exit 1; }
    name=$tag-$mtp-$arm
    out=$RUN/serve-$name
    echo "== $(date +%H:%M:%S) $name cfg={$cfg} ${kv:-}"
    mkdir -p "$out"
    # clock / power / temperature, 1 Hz, for the life of the serve
    ( while true; do
          echo "## $(date +%s.%N)"
          rocm-smi -d $SMI_DEVS --showclocks --showpower --showuse 2>/dev/null \
              | grep -E "sclk|mclk|Power \(|GPU use"
          sleep 1
      done ) > "$out/clocks.log" 2>&1 &
    clk=$!
    echo "$clk" > "$out/clocks.pid"
    ARM=$arm RECIPE=flashnext-$mtp TAG=$name MONO_CFG="$cfg" \
        CACHE=$RUN/cache-2a-$tag-$arm CELLS="$CELLS" BENCH_EXTRA="--temperature 0" \
        LOAD_HOOK=${LOAD_HOOK:-$HERE/parity_hook.sh} \
        bash "$HERE/serve_2a.sh" ${kv:-} > "$RUN/$name.out" 2>&1
    kill "$clk" 2>/dev/null; rm -f "$out/clocks.pid"
    zcat -f "$out"/serve.log* 2>/dev/null | grep -a "rdna_mono_ffn timing" \
        | sed 's/^.*rdna_mono_ffn timing //' \
        | awk '{k=$1" "$2; last[k]=$0} END{for (k in last) print last[k]}' | sort > "$out/timing.txt"
    grep -E "READY|FAILED|DIED|PROBE|PPL|PREFIX|CONCURRENCY|RESULT|bench-" "$out/summary.txt" \
        | grep -vE "^\s+(PASS|FAIL)  prompt" | cut -c1-200
    grep -E "^dev0 " "$out/timing.txt"
done
