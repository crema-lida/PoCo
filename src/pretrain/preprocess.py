"""
Preprocess raw polymer SMILES dataset.

Purpose
-------
Clean the raw dataset and ensure each SMILES string is valid.
Writes valid lines to OUTPUT_PATH, logs invalid lines to *.failed.csv,
and persists restartable progress to *.log (as PROCESSED=<n>).
"""

import multiprocessing as mp
from pathlib import Path
from itertools import islice
from tqdm import tqdm
from psmiles import PolymerSmiles as PS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_PATH  = Path("./datasets/OMG/OMG_polymers.csv")
OUTPUT_PATH = Path("./datasets/OMG/cleaned/OMG_polymers.csv")

# I/O tuning ----------------------------------------------------------------
BATCH_LINES    = 8192          # flush after this many output lines
BUFFER_BYTES   = 4 << 20       # 4 MiB

# Parallelism ---------------------------------------------------------------
N_PROC         = 24
CHUNK_SIZE     = 256


if __name__ == "__main__":
    FAILED_PATH = OUTPUT_PATH.parent / "failed.csv"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Stream remaining lines only
    def iterate_remaining(path: Path, skip: int = 0):
        with path.open() as fh:
            for idx, ln in enumerate(islice(fh, skip, None), start=1):
                ln = ln.strip()
                ln = ln.split(',')[-1]
                if ln:
                    yield idx, ln

    # Validate one line
    def _process(job: tuple[int, str]):
        idx, raw = job
        try:
            PS(raw)
            return idx, raw, None
        except Exception as e:
            return idx, None, f"{idx}\t{raw}\t{type(e).__name__}: {e}\n"

    raw_iter = iterate_remaining(INPUT_PATH, 1)

    next_to_write = 1
    buffer: dict[int, str] = {}     # hold successful lines by index (to write in order)
    fail_idx: set[int] = set()      # failed indices to advance the committed prefix
    pending_ok: list[str] = []
    pending_fail: list[str] = []

    with mp.Pool(N_PROC) as pool, \
         OUTPUT_PATH.open("a", buffering=BUFFER_BYTES) as fh, \
         FAILED_PATH.open("a", buffering=BUFFER_BYTES) as ffail:

        for idx, out_line, fail_line in tqdm(
            pool.imap_unordered(_process, raw_iter, chunksize=CHUNK_SIZE),
            desc="Processing", unit="lines",
        ):
            if out_line is not None:
                buffer[idx] = out_line
            if fail_line is not None:
                fail_idx.add(idx)
                pending_fail.append(fail_line)
                if len(pending_fail) >= BATCH_LINES:
                    ffail.writelines(pending_fail)
                    pending_fail.clear()

            # Commit ordered prefix: write successes or skip failures
            while True:
                if next_to_write in buffer:
                    pending_ok.append(buffer.pop(next_to_write))
                    next_to_write += 1
                    if len(pending_ok) >= BATCH_LINES:
                        fh.writelines(line + "\n" for line in pending_ok)
                        pending_ok.clear()
                elif next_to_write in fail_idx:
                    next_to_write += 1
                else:
                    break

        # Final flush
        if pending_ok:
            fh.writelines(line + "\n" for line in pending_ok)
        if pending_fail:
            ffail.writelines(pending_fail)

    print("[DONE] All SMILES processed and saved.")
