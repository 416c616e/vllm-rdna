# MonoKernel P0 probe (V620 / gfx1030, ROCm 7.14.0)

> Drafter: ROCM_specialist.
>
> **Not yet compiled; first build on V620 with ROCm 7.14 may need small fixes.**


Standalone HIP probe for step 1 (P0) on card PVTI_lAHOAOnQn84BgjK3zhAaHOk. It makes no fork changes and has no vLLM dependency.

    ./build.sh          # builds build/wgp/probe (default WGP mode) and build/cumode/probe (-mcumode)
    ./run.sh            # env + versions + both variants -> logs/p0-<timestamp>.log
    ./run.sh --lds 32768 --replays 500 --busy-ms 2000   # extra args go to the probe

Flags: `--offload-arch=gfx1030 -O3 -ffp-contract=fast -std=c++17`, no fast-math. Override paths with `ROCM_PATH`, `HIPCC` or `READELF`.

## What it tests
* **Kernel:** 72 WGs x 256 threads, each with dynamic LDS set by `--lds` (default: 65536 minus the kernel's static LDS, rounded down to 16 B), touched every phase. In each phase, WG w stores tag p+1 to `flags[w]` (release, agent scope) and waits for `flags[(w+1)%72] >= p+1` (acquire, agent scope). A memset captured in the same graph zeroes the flags before every launch.
* **[residency]:** `hipOccupancyMaxActiveBlocksPerMultiprocessor` x CU count for both kernels, plus `hipFuncGetAttributes`. The *real* numbers come from the code object, printed by run.sh: `kernel_meta.txt` (`.sgpr_count`, `.vgpr_count`, `.group_segment_fixed_size`, spills) and `amdhsa_directives.txt` (`.amdhsa_next_free_vgpr/sgpr`, `.amdhsa_float_denorm_mode_32/16_64`, `.amdhsa_workgroup_processor_mode`, which should be 1 for wgp and 0 for cumode).
* **[direct] / [graph]:** a direct launch, then stream capture, instantiate and N replays for the plain launch and for `hipLaunchCooperativeKernel` (adds `cg::this_grid().sync()` each phase). Each capture, instantiate and replay step prints its result or the HIP error.
* **[progress]:** a busy kernel (CUs x 4 WGs, `--busy-ms`) on stream B and the flag kernel on stream A, launched in the order set by `--order` (see below). Waits are bounded with `wall_clock64()` (s_memrealtime at a fixed rate; ticks come from `hipDeviceAttributeWallClockRate`, `--wait-ms`). On timeout the first WG writes `{phase, wg, peer, observed, expected}` to a sticky record in coherent pinned host memory, and every WG bails. The host polls with `hipStreamQuery` under `--host-ms`, so it never blocks forever. Output is `PASS` or `STALL REPORTED phase=.. wg=..`.

## Forward-progress order
`--order busy-first|flag-first` (default busy-first), with `--gap-ms` (default 20) as a host sleep between the two launches so the GPU sees them in that order. The log records the order and the host launch and completion timestamps. run.sh runs both orders for both builds.

## Reading results
* **Plain launch stalls only with busy-first, and flag-first passes:** the protocol is fine. The flag kernel's workgroups couldn't all become resident while another kernel held the CUs. Step 2a therefore needs exclusive use of the GPU while it runs (or a cooperative launch), not a protocol fix.
* **Stall with flag-first, or a coop stall in either order:** suspect the protocol, the residency budget or ROCm. Check the reported phase and WG and the code-object LDS and VGPR numbers.
* *Expected, unmeasured (no V620 run yet):* with `--lds 65536`, 1 block/CU in cumode, and in WGP mode 2 per WGP (128 KiB), so 72 should still be resident. If the occupancy result x CUs is below 72, the full residency gate fails.
* A progress STALL with the busy kernel present means a WG isn't co-resident or isn't scheduled. This is the deadlock class the MonoKernel has to rule out. It's expected for the plain launch; the cooperative launch should be guaranteed co-resident.

## LDS budget
gfx1030 allows 64 KiB of LDS per workgroup, and that covers static **plus** dynamic LDS. `flag_kernel` has static `__shared__` state (`bail`). The probe reads its size with `hipFuncGetAttributes().sharedSizeBytes`. Without `--lds`, the dynamic size is `65536 - static`, rounded down to 16 B. If you pass `--lds` and static + dynamic is over 65536, the probe exits with an error before any capture or launch. The config line prints the static, dynamic and total sizes. The code-object value (`.group_segment_fixed_size` in `kernel_meta.txt` / `amdhsa_directives.txt`) is **unmeasured until the first V620 build**.

## Caveats
* `grid.sync()` is not bounded. If a WG bails in the coop variant, the others hang in grid.sync. The host timeout catches it, the sticky record still names the phase, and the process exits with code 3 without teardown. run.sh also wraps each run in `timeout -s KILL 300`.
* Wait bounds use `wall_clock64()` (s_memrealtime, fixed rate), not `clock64()` (s_memtime, which follows the shader clock and drifts with DPM).
* `--save-temps` file names vary between hipcc versions. If `kernel_meta.txt` is missing, run `roc-obj-ls build/<v>/probe` / `roc-obj -d` and `llvm-readelf --notes` on the extracted `.co`.
* Not compiled or run here: there was no ROCm on the drafting box.
