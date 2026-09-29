# A listing of the Codex branch changes

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
