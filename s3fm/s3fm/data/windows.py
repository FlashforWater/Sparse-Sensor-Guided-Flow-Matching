"""Window extraction with the canonical tensor convention.

Spec tensor convention for KSE:  [batch, T, C, Nx]
  - batch : number of extracted windows
  - T     : window_length, number of physical frames (frame_idx / n)
  - C     : channels / QoIs (KSE has C=1)
  - Nx    : spatial grid points

Every window carries its global physical frame indices so that overlap logic in
later milestones can assert two copies of a frame refer to the same global index
(the "time semantics" correctness rule). We never collapse frame_idx (physical
time, n) and flow_time (s) into one variable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowBatch:
    """A batch of windows plus their physical-frame bookkeeping.

    ``data`` has shape ``[batch, T, C, Nx]``. ``frame_idx`` has shape
    ``[batch, T]`` giving, for each window and each in-window position, the
    global physical frame index in the source trajectory. ``traj_idx`` has shape
    ``[batch]`` identifying the source trajectory of each window.
    """

    data: np.ndarray          # [batch, T, C, Nx]
    frame_idx: np.ndarray     # [batch, T]   global physical frame indices
    traj_idx: np.ndarray      # [batch]      source trajectory id
    physical_dt: float        # spacing between consecutive frames

    def __post_init__(self):
        b, T, C, Nx = self.data.shape
        if self.frame_idx.shape != (b, T):
            raise ValueError(f"frame_idx shape {self.frame_idx.shape} != ({b},{T})")
        if self.traj_idx.shape != (b,):
            raise ValueError(f"traj_idx shape {self.traj_idx.shape} != ({b},)")

    @property
    def shape(self):
        return self.data.shape


def add_channel_axis(trajectories: np.ndarray) -> np.ndarray:
    """Turn ``(n_traj, n_steps, Nx)`` into ``(n_traj, n_steps, C=1, Nx)``."""
    if trajectories.ndim != 3:
        raise ValueError(f"expected (n_traj, n_steps, Nx), got {trajectories.shape}")
    return trajectories[:, :, None, :]


def extract_windows(
    trajectories: np.ndarray,
    traj_ids: tuple[int, ...] | list[int],
    window_length: int,
    stride: int,
    physical_dt: float,
) -> WindowBatch:
    """Slide a window of length ``window_length`` along each trajectory.

    ``trajectories`` has shape ``(n_traj, n_steps, C, Nx)`` (channel axis present;
    use :func:`add_channel_axis` first). ``traj_ids`` gives the *global* id of
    each trajectory row, so extracted windows record the right ``traj_idx`` even
    when called on a single split (where row order is not the global order).

    Windows that would run past the end of a trajectory are dropped (no padding).
    """
    if trajectories.ndim != 4:
        raise ValueError(f"expected (n_traj, n_steps, C, Nx), got {trajectories.shape}")
    n_traj, n_steps, C, Nx = trajectories.shape
    if len(traj_ids) != n_traj:
        raise ValueError("traj_ids length must match number of trajectory rows")
    if window_length > n_steps:
        raise ValueError("window_length exceeds trajectory length")

    data_chunks, frame_chunks, traj_chunks = [], [], []
    starts = range(0, n_steps - window_length + 1, stride)
    for row in range(n_traj):
        for s in starts:
            data_chunks.append(trajectories[row, s : s + window_length])  # [T,C,Nx]
            frame_chunks.append(np.arange(s, s + window_length))           # [T]
            traj_chunks.append(traj_ids[row])

    data = np.stack(data_chunks, axis=0)
    frame_idx = np.stack(frame_chunks, axis=0)
    traj_idx = np.asarray(traj_chunks, dtype=np.int64)
    return WindowBatch(data=data, frame_idx=frame_idx, traj_idx=traj_idx, physical_dt=physical_dt)
