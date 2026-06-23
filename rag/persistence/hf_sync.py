"""Persist the local ``data/`` directory to a private HuggingFace dataset.

Ephemeral hosts (HF Spaces, scale-to-zero containers) lose the FAISS index,
uploads, registry, users, and job history on restart. When ``HF_TOKEN`` and
``HF_DATASET_REPO`` are set, the app pulls that state on startup and pushes it
back after each mutation (ingestion / deletion) and on shutdown, giving free,
durable persistence. Disabled (and a complete no-op) when the vars are unset, so
local and VPS deployments are unaffected.

All operations are best-effort: a sync failure is logged but never breaks a
request or startup.
"""

from __future__ import annotations

import logging
import threading

from app.config import get_settings

logger = logging.getLogger("docmind")

_push_lock = threading.Lock()


def enabled() -> bool:
    return get_settings().has_hf_sync


def pull() -> None:
    """Download the persisted dataset into the local data dir (startup)."""
    if not enabled():
        return
    s = get_settings()
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=s.hf_dataset_repo,
            repo_type="dataset",
            token=s.hf_token,
            local_dir=s.data_dir,
        )
        logger.info("Pulled persisted data from dataset %s", s.hf_dataset_repo)
    except Exception as exc:  # noqa: BLE001 — first run / network / missing repo
        logger.info("HF data pull skipped (%s): %s", s.hf_dataset_repo, exc)


def push(reason: str = "sync") -> None:
    """Upload the local data dir back to the dataset (after a mutation)."""
    if not enabled():
        return
    s = get_settings()
    if not _push_lock.acquire(blocking=False):
        return  # a push is already in flight; its snapshot will include our change
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=s.hf_token)
        api.create_repo(
            repo_id=s.hf_dataset_repo, repo_type="dataset", private=True, exist_ok=True
        )
        api.upload_folder(
            folder_path=s.data_dir,
            repo_id=s.hf_dataset_repo,
            repo_type="dataset",
            commit_message=f"DocMind data sync: {reason}",
            ignore_patterns=["**/.cache/**", ".cache/**"],
        )
        logger.info("Pushed data to dataset %s (%s)", s.hf_dataset_repo, reason)
    except Exception as exc:  # noqa: BLE001 — best-effort persistence
        logger.warning("HF data push failed: %s", exc)
    finally:
        _push_lock.release()


def push_async(reason: str = "sync") -> None:
    """Fire-and-forget push so callers (e.g. the ingest worker) don't block."""
    if not enabled():
        return
    threading.Thread(target=push, args=(reason,), daemon=True).start()
