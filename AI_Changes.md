# A combined listing of the brent_factors_logging and merged parent branch changes

## 1. Parameterization 3 stage-1 continuation

Commit: `967e5ea`

| Modified source file | Brief description |
| --- | --- |
| `ecm.c` | Allow CPU continuation from parameterization 3 saves to a higher B1, preserving the saved curve and residue, including a zero residue. |

## 2. Parameterization 3 with 64-bit sigmas

Commits: `dafab73`, `0370d52`, `4680298`, `9b37c93`, `89cdd15`

The denominator remains `2^32`. The 33-bit headroom optimization applies only to wide sigma batches and does not increase the selected kernel size.

| Modified source file | Brief description |
| --- | --- |
| `batch.c` | Use the existing general-coefficient CPU ladder when a parameterization 3 coefficient exceeds one limb. |
| `cgbn_stage1.cu` | Support 64-bit sigmas, split coefficient multiplication into 32-bit parts, and defer reductions when 33 spare bits are available. |
| `cgbn_stage1.h` | Widen the internal GPU stage-1 sigma argument to `uint64_t`. |
| `cudawrapper.c` | Generate and validate 64-bit GPU sigma ranges; preserve full values during curve checks, stage 2, and factor reporting. |
| `parametrizations.c` | Generate 64-bit random sigmas for parameterization 3. |

## 3. GWNUM buffer allocation fix

Commit: `1f811a0`

| Modified source file | Brief description |
| --- | --- |
| `Fgw.c` | Use `GMP_NUMB_BITS` in both residue-buffer calculations instead of the size of the limb counter, preventing underallocation on Windows. |

## 4. Brent metadata support

Commit: `89cdc9d`

| Modified source file | Brief description |
| --- | --- |
| `ecm-ecm.h` | Add a private candidate label and internal parsing/display declarations. |
| `candi.c` | Initialize and clear labels when candidates change; retain them when reducing a cofactor. |
| `eval.c` | Detect Brent rows and parse the base, exponent, sign, optional L/M suffix, and composite. |
| `auxi.c` | Display the source number with factor reports and control Brent detection when reading input. |
| `main.c` | Display input metadata and pass a verified special-form parent to GWNUM, selecting Suyama automatically when eligible. |
| `resume.c` | Save and restore metadata through a single backward-compatible `COMMENT=LABEL=...` field; keep Brent parsing disabled for `N`. |

## 5. Binary file modes for Lucas-chain data

Commit: `f2ec9a5`

| Modified source file | Brief description |
| --- | --- |
| `ecm.c` | Open the CPU stage-1 Lucas-chain file with `rb`. |
| `gw_ecmstag1.c` | Open the GWNUM stage-1 Lucas-chain file with `rb`. |
| `LucasChainGenerator/src/LucasChainGen.c` | Change eleven file-opening modes to `rb`, `wb`, or `ab` for chain records, pending work, and restart data. |

## 6. GWNUM support for parameterization 3

Commit: `902db13`

| Modified source file | Brief description |
| --- | --- |
| `ecm.c` | Allow GWNUM stage 1 for parameterization 3; avoid repeating completed work or replacing its residue, and compute the batch exponent only when needed. |
| `gw_ecmstag1.c` | Finish or interrupt stage 1 without an extra doubling when continuation requires no new power of two, in both the 32-bit and 64-bit interfaces. |
| `main.c` | Compute the batch exponent when `-bsaves` requests it, even if GWNUM handled stage 1. |

Follow-up commit `a18ad3c` restores non-executable file permissions on these three source files.

## 7. Combined GPU and GWNUM build fixes

Commit: `c4cfe49`

| Modified source file | Brief description |
| --- | --- |
| `Fgw.c` | Remove substitute C++ exception and static-initialization guard functions so the real runtime supplies them. |
| `configure.ac` | Link GWNUM with `libstdc++` as well as pthreads. |
| `acinclude.m4` | Use `CPPFLAGS` for the CGBN header check and keep GMP linker flags out of CUDA compiler flags. |

## 8. MSVC x64 performance improvements

Commit: `4ba74da`

| Modified source file | Brief description |
| --- | --- |
| `longlong.h` | Use `_umul128` for full 64-by-64-bit multiplication under MSVC x64, accelerating arithmetic used in stage 2. |
| `ecm-params.h` | Select the x86-64 tuning parameters for native MSVC x64 builds; exclude ARM64EC. |

## 9. Brent factor logging with optional external group orders

| Modified source file | Brief description |
| --- | --- |
| `brent_log.c` | Append locked, single-line factor records; optionally run the separately supplied Windows/Linux grouporder helper and capture its results. |
| `auxi.c`, `main.c`, `ecm-ecm.h` | Log proper Brent factors, including initialization factors, and echo the exact entry while preserving quiet stdout. |
| `ecm.h.in`, `factor.c`, `ecm.c`, `pm1.c`, `pp1.c` | Return selected curve and bound information for accurate logging. |
| `cudawrapper.c` | Preserve originating sigmas and stage numbers when GPU factors are reduced and sorted. |
| `Makefile.am`, `build.vs/{ecm,ecm_gpu,multiecm}/*.vcxproj*` | Include CLI logging in Linux and Windows builds; distribute focused tests. |
| `tests/test-brent-logging.py`, `tests/grouporder-stub.c` | Check real helpers, missing/failed helpers, metadata, bounds, GPU attribution, and concurrent appends. |
| `README`, `.gitignore` | Document the external helper contract and ignore local validation build artifacts. |

## 10. Windows Lucas-chain file lookup

Commit: `a72a165`

| Modified source file | Brief description |
| --- | --- |
| `auxlib.c`, `lchain.h` | Share binary file lookup: working directory first, then beside the executable on Windows if the local file is absent; support Unicode executable paths. |
| `ecm.c`, `gw_ecmstag1.c` | Use the shared lookup for CPU ECM and GWNUM, retaining PRAC fallback without changing the working directory. |
| `Makefile.am` | Include the new header and lookup regression test in source distributions. |
| `tests/test-lchain-path.py` | Check lookup precedence, missing files, local open errors, Unicode paths, PATH invocation, and unchanged residues on Windows. |


