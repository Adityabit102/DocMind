"""Fetch Donut (donut-base-finetuned-cord-v2) weights into ``models/donut-cord/``.

Why this exists: the ``huggingface_hub`` downloader (both the default backend and
``hf_transfer``) reproducibly wedges on the 806 MB weights blob over anonymous
connections — it stalls at a fixed byte offset and never resumes. A plain HTTP
range request to the *same* URL, however, streams fine (~20+ MB/s). So this
script streams the weights directly with a resumable loop and drops them next to
the small config/tokenizer files, giving a local model dir that
``document_extraction._default_donut_model()`` picks up automatically.

Usage:
    python -m vlm_module.fetch_donut

Idempotent: re-running resumes a partial file and no-ops once complete.
"""

from __future__ import annotations

import os
import shutil
import time
import urllib.request
from pathlib import Path

REPO = "naver-clova-ix/donut-base-finetuned-cord-v2"
BASE = f"https://huggingface.co/{REPO}/resolve/main"
SMALL_FILES = [
    "config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json",
    "preprocessor_config.json", "sentencepiece.bpe.model",
]
WEIGHTS = "pytorch_model.bin"
DEST = Path(__file__).resolve().parent.parent / "models" / "donut-cord"


def _download(url: str, dst: Path, resumable: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    total = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD"),
                                       timeout=30).headers["Content-Length"])
    if dst.exists() and dst.stat().st_size >= total:
        print(f"  {dst.name}: already complete ({total/1e6:.1f} MB)")
        return
    attempt = 0
    while True:
        have = dst.stat().st_size if dst.exists() else 0
        if have >= total:
            break
        attempt += 1
        if attempt > 60:
            raise RuntimeError(f"{dst.name}: gave up after 60 resume attempts")
        headers = {"Range": f"bytes={have}-"} if resumable and have else {}
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
            mode = "ab" if (resumable and have) else "wb"
            t0, last = time.time(), have
            with open(dst, mode) as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    now = dst.stat().st_size
                    if now - last >= 100 * (1 << 20):
                        print(f"  {dst.name}: {now/1e6:.0f}/{total/1e6:.0f} MB "
                              f"({(now-have)/1e6/max(time.time()-t0,0.1):.1f} MB/s)")
                        last = now
        except Exception as exc:  # noqa: BLE001 — resume on any transport hiccup
            print(f"  {dst.name}: stall/retry #{attempt} ({type(exc).__name__}); resuming")
            time.sleep(2)


def main() -> None:
    print(f"Fetching Donut into {DEST}")
    for name in SMALL_FILES:
        _download(f"{BASE}/{name}", DEST / name)
    print("Fetching weights (streamed, resumable) ...")
    _download(f"{BASE}/{WEIGHTS}", DEST / WEIGHTS, resumable=True)
    size = (DEST / WEIGHTS).stat().st_size
    print(f"Done. Weights = {size/1e6:.1f} MB. Donut engine will load from {DEST}.")


if __name__ == "__main__":
    main()
