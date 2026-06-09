"""Runtime LMDB chunk dataset.

Ports OLD ``scripts/lmdb_dataset.py::LMDBChunkDataset`` — the read side of the LMDB contract written
by ``lmdb_writer`` (1.2). One instance wraps one ``*.lmdb`` chunk; the chunk loader (4.2) opens one
per chunk. The multiprocessing-correctness core is preserved verbatim:

* **per-process env** (:meth:`_get_env`, pid-keyed) — a single ``lmdb.Environment`` is not safe to share
  across forked/spawned workers, so each process lazily opens its own and reopens if the pid changes.
* **picklable** (:meth:`__getstate__` drops ``_env``/``_pid``) — DataLoader ``spawn``/``fork`` pickles the
  dataset to each worker; the live handle must not travel.

Behavior preserved vs OLD: lexicographic ``_meta`` cursor order for ``seq_ids``, per-frame JPEG decode,
and the hard frame-count-mismatch error (a corrupt chunk fails loudly, never silently short). Read-time
ImageNet normalize lives in the injected transforms (:func:`build_read_transforms`), never in the writer.
"""

from __future__ import annotations

import io
import os
import pickle
from collections.abc import Callable
from pathlib import Path

import lmdb
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from pedpredict.config.schema import DataCfg
from pedpredict.data.transforms import build_read_transforms

__all__ = ["LMDBChunkDataset"]


class LMDBChunkDataset(Dataset):
    """Worker-safe ``Dataset`` over one LMDB chunk; yields per-sequence tight/context/motions + labels."""

    def __init__(
        self,
        lmdb_path: str | Path,
        transform_tight: Callable[[Image.Image], Tensor],
        transform_context: Callable[[Image.Image], Tensor],
    ) -> None:
        self.lmdb_path = str(lmdb_path)
        self.transform_tight = transform_tight
        self.transform_context = transform_context
        self._env: lmdb.Environment | None = None
        self._pid: int | None = None

        # Index the chunk's sequence ids in LMDB cursor (lexicographic) order — OLD parity.
        self.seq_ids: list[str] = []
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        try:
            with env.begin(write=False) as txn:
                for key, _ in txn.cursor():
                    key_str = key.decode()
                    if key_str.endswith("_meta"):
                        self.seq_ids.append(key_str.split("_")[0])
        finally:
            env.close()
        print(f"[LMDBChunkDataset] Loaded index from {self.lmdb_path}: {len(self.seq_ids)} sequences")

    @classmethod
    def from_config(cls, lmdb_path: str | Path, cfg: DataCfg) -> LMDBChunkDataset:
        """Build with config-driven read transforms (Resize -> ToTensor -> ImageNet Normalize)."""
        transform_tight, transform_context = build_read_transforms(cfg)
        return cls(lmdb_path, transform_tight, transform_context)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_env"] = None   # live handle must not cross the worker pickle boundary
        state["_pid"] = None
        return state

    def __del__(self) -> None:
        if self._env is not None:
            self._env.close()

    def _get_env(self) -> lmdb.Environment:
        pid = os.getpid()
        if self._env is None or self._pid != pid:
            if self._env is not None:
                self._env.close()
            self._env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
            self._pid = pid
        return self._env

    def __len__(self) -> int:
        return len(self.seq_ids)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        seq_id = self.seq_ids[idx]
        env = self._get_env()
        with env.begin(write=False) as txn:
            meta = pickle.loads(txn.get(f"{seq_id}_meta".encode()))
            motions, actions = meta["motions"], meta["actions"]
            looks, crosses = meta["looks"], meta["crosses"]

            t_frames = motions.shape[0]   # frame count comes from the motion tensor (1.2 contract)
            imgs_tight, imgs_context = [], []
            for k in range(t_frames):
                tbuf = txn.get(f"{seq_id}_{k}_tight".encode())
                cbuf = txn.get(f"{seq_id}_{k}_context".encode())
                if tbuf is None or cbuf is None:
                    continue
                timg = Image.open(io.BytesIO(tbuf)).convert("RGB")
                cimg = Image.open(io.BytesIO(cbuf)).convert("RGB")
                imgs_tight.append(self.transform_tight(timg))
                imgs_context.append(self.transform_context(cimg))

            if len(imgs_tight) != t_frames:
                raise ValueError(
                    f"[LMDBChunkDataset] Sequence {seq_id!r}: expected {t_frames} frames, "
                    f"found {len(imgs_tight)} — missing LMDB frame keys. "
                    f"Chunk may be corrupted: {self.lmdb_path}"
                )

        return {
            "images_tight": torch.stack(imgs_tight),
            "images_context": torch.stack(imgs_context),
            "motions": motions,
            "actions": actions,
            "looks": looks,
            "crosses": crosses,
        }
