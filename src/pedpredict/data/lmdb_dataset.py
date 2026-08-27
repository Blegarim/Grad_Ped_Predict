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
import logging
import os
import pickle
import random
from collections.abc import Callable
from pathlib import Path

import lmdb
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from pedpredict.config.schema import DataCfg
from pedpredict.data.pie_sequences import ONSET_FIELDS
from pedpredict.data.transforms import (
    ProcessedSample,
    build_read_transforms,
    imagenet_normalize,
    resize_to_tensor,
)

__all__ = ["LMDBChunkDataset", "read_raw_sample"]

#: Decode a stored crop to an un-normalized ``[0, 1]`` CHW tensor at its STORED size (no resize, no
#: ImageNet normalize) — the inverse of ``lmdb_writer.encode_jpeg_bytes``. Built once (stateless).
_DECODE_TO_TENSOR = transforms.ToTensor()


def read_raw_sample(txn: lmdb.Transaction, seq_id: str, lmdb_path: str = "") -> ProcessedSample:
    """Decode one stored sample back into an un-normalized :class:`ProcessedSample` — the inverse of
    :func:`pedpredict.data.lmdb_writer.write_sample`.

    Crops are decoded to ``[0, 1]`` CHW at their STORED size (no resize, no normalize) and the full
    ``MOTION_STORE_DIM`` motion vector + ``track_id``/``tte`` are kept, so the result is the writer's
    input modulo JPEG. Offline augmentation uses this to re-augment the built train LMDB after the
    incremental build deletes the source frames; the runtime :class:`LMDBChunkDataset` read path instead
    resizes + normalizes + slices motions (lossy for re-encoding), so it must not be reused here.
    """
    meta = pickle.loads(txn.get(f"{seq_id}_meta".encode()))
    if "track_id" not in meta:
        raise ValueError(
            f"[read_raw_sample] Sequence {seq_id!r} has v1 meta (no track_id) — rebuild required: {lmdb_path}"
        )
    motions = meta["motions"]
    t_frames = motions.shape[0]
    imgs_tight, imgs_context = [], []
    for k in range(t_frames):
        tbuf = txn.get(f"{seq_id}_{k}_tight".encode())
        cbuf = txn.get(f"{seq_id}_{k}_context".encode())
        if tbuf is None or cbuf is None:
            raise ValueError(
                f"[read_raw_sample] Sequence {seq_id!r}: missing frame {k} blob — corrupt chunk: {lmdb_path}"
            )
        imgs_tight.append(_DECODE_TO_TENSOR(Image.open(io.BytesIO(tbuf)).convert("RGB")))
        imgs_context.append(_DECODE_TO_TENSOR(Image.open(io.BytesIO(cbuf)).convert("RGB")))
    return ProcessedSample(
        images_tight=torch.stack(imgs_tight),
        images_context=torch.stack(imgs_context),
        motions=motions,
        actions=torch.as_tensor(meta["actions"], dtype=torch.long),
        looks=torch.as_tensor(meta["looks"], dtype=torch.long),
        crosses=torch.as_tensor(meta["crosses"], dtype=torch.long),
        track_id=meta["track_id"],
        tte=meta.get("tte"),
        pose=meta.get("pose"),
        # S1: carried verbatim so an offline re-augment round-trip (augment_dataset.py) preserves them.
        **{key: meta.get(key) for key in ONSET_FIELDS},
    )

# Q6: per-open chatter goes through logging at DEBUG (the dataset is re-opened per chunk per epoch —
# printed, it buried real warnings like the C1 chunk skips). Enable via logging config when needed.
_LOGGER = logging.getLogger(__name__)


class LMDBChunkDataset(Dataset):
    """Worker-safe ``Dataset`` over one LMDB chunk; yields per-sequence tight/context/motions + labels."""

    def __init__(
        self,
        lmdb_path: str | Path,
        transform_tight: Callable[[Image.Image], Tensor],
        transform_context: Callable[[Image.Image], Tensor],
        motion_dim: int | None = None,
        *,
        augmentor: Callable[[ProcessedSample, random.Random], ProcessedSample] | None = None,
        aug_seed: int = 0,
        normalize: Callable[[Tensor], Tensor] | None = None,
        pose_transform: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> None:
        self.lmdb_path = str(lmdb_path)
        self.transform_tight = transform_tight
        self.transform_context = transform_context
        # v2 store-wide/slice-narrow contract (A4/M9): the chunk stores MOTION_STORE_DIM channels;
        # __getitem__ slices to this consumed width. None -> return the full stored vector.
        self.motion_dim = motion_dim
        # Runtime augmentation (train split only): transforms decode to un-normalized [0, 1] and
        # `normalize` is applied AFTER `augmentor` (color jitter needs un-normalized crops).
        self._augmentor = augmentor
        self._aug_seed = aug_seed
        self._normalize = normalize
        # Pose arm (docs/POSE_ENCODER.md): (pose [T,23,3], motions [T,9]) -> [T, 9+dim] "motions".
        # Applied AFTER the augmentor (flip mutates raw pose + motions); implies motion_dim=None.
        self._pose_transform = pose_transform
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
        _LOGGER.debug("Loaded index from %s: %d sequences", self.lmdb_path, len(self.seq_ids))

    @classmethod
    def from_config(
        cls,
        lmdb_path: str | Path,
        cfg: DataCfg,
        *,
        augmentor: Callable[[ProcessedSample, random.Random], ProcessedSample] | None = None,
        aug_seed: int = 0,
        pose_transform: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> LMDBChunkDataset:
        """Build with config-driven read transforms + the ``cfg.motion_dim`` consumed-width slice.

        With an ``augmentor`` (train split only) frames decode to un-normalized ``[0, 1]`` and the
        ImageNet norm is deferred to after augmentation; without one the read path is unchanged.
        With a ``pose_transform`` (``pose_motion_transform(cfg)``, pose.enabled) the stored motions
        are kept full-width and replaced by the built ``[T, 9 + pose]`` vector.
        """
        motion_dim = None if pose_transform is not None else cfg.motion_dim
        if augmentor is None:
            transform_tight, transform_context = build_read_transforms(cfg)
            return cls(
                lmdb_path, transform_tight, transform_context, motion_dim=motion_dim,
                pose_transform=pose_transform,
            )
        transform_tight = resize_to_tensor((cfg.img_height, cfg.img_width))
        transform_context = resize_to_tensor((cfg.read_context_height, cfg.read_context_width))
        return cls(
            lmdb_path, transform_tight, transform_context, motion_dim=motion_dim,
            augmentor=augmentor, aug_seed=aug_seed, normalize=imagenet_normalize(cfg),
            pose_transform=pose_transform,
        )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_env"] = None   # live handle must not cross the worker pickle boundary
        state["_pid"] = None
        return state

    def close(self) -> None:
        """Release this process's LMDB handle (idempotent; the next read lazily reopens).

        Windows refuses to open a file for write while a memory-mapped section is live, so any tool
        that reopens a chunk for writing — ``scripts/backfill_onset_meta.py`` above all — must see
        every reader closed first. Explicit beats relying on ``__del__`` and GC timing.
        """
        if self._env is not None:
            self._env.close()
            self._env = None
            self._pid = None

    def __del__(self) -> None:
        self.close()

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

    def __getitem__(self, idx: int) -> dict[str, Tensor | str | int]:
        seq_id = self.seq_ids[idx]
        env = self._get_env()
        with env.begin(write=False) as txn:
            meta = pickle.loads(txn.get(f"{seq_id}_meta".encode()))
            motions, actions = meta["motions"], meta["actions"]
            looks, crosses = meta["looks"], meta["crosses"]
            if "track_id" not in meta:
                raise ValueError(
                    f"[LMDBChunkDataset] Sequence {seq_id!r} has v1 meta (no track_id) — this chunk "
                    f"predates the v2 rebuild and must be rebuilt: {self.lmdb_path}"
                )
            track_id = meta["track_id"]
            pose = None
            if self._pose_transform is not None:
                pose = meta.get("pose")
                if pose is None:
                    raise ValueError(
                        f"[LMDBChunkDataset] Sequence {seq_id!r} has no pose meta — this chunk was "
                        f"built without pose.enabled and must be rebuilt: {self.lmdb_path}"
                    )
            if self.motion_dim is not None:
                if motions.shape[1] < self.motion_dim:
                    raise ValueError(
                        f"[LMDBChunkDataset] Sequence {seq_id!r}: stored motions have "
                        f"{motions.shape[1]} channels < motion_dim={self.motion_dim} — stale chunk, "
                        f"rebuild required: {self.lmdb_path}"
                    )
                motions = motions[:, : self.motion_dim]

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

        tight, context = torch.stack(imgs_tight), torch.stack(imgs_context)
        if self._augmentor is not None:
            assert self._normalize is not None  # from_config pairs augmentor with the deferred norm
            ps = ProcessedSample(
                images_tight=tight, images_context=context, motions=motions,
                actions=torch.as_tensor(actions), looks=torch.as_tensor(looks),
                crosses=torch.as_tensor(crosses), track_id=track_id, tte=meta.get("tte"),
                pose=pose,
                # S1 labels describe the FUTURE, so no augmentation can alter them — carried
                # through only so `ps` stays a faithful sample for any augmentor that inspects it.
                **{key: meta.get(key) for key in ONSET_FIELDS},
            )
            ps = self._augmentor(ps, random.Random(self._aug_seed * 1_000_003 + idx))
            tight = self._normalize(ps.images_tight)
            context = self._normalize(ps.images_context)
            motions = ps.motions
            pose = ps.pose
        if self._pose_transform is not None:
            motions = self._pose_transform(pose, motions)

        sample: dict[str, Tensor | str | int] = {
            "images_tight": tight,
            "images_context": context,
            "motions": motions,
            "actions": actions,
            "looks": looks,
            "crosses": crosses,
            "track_id": track_id,  # str — eval-side track aggregation (M6); collate ignores it
        }
        if "tte" in meta:  # M5 benchmark-protocol chunks only
            sample["tte"] = meta["tte"]
        # S1 onset annotation — present only in S1-annotated builds (or after the backfill script).
        # Emitted as 0-dim long tensors so `collate_sequences` can stack them like any other label;
        # all-or-nothing per chunk, matching how the writer packs them.
        for key in ONSET_FIELDS:
            if key in meta:
                sample[key] = torch.as_tensor(meta[key], dtype=torch.long)
        return sample
