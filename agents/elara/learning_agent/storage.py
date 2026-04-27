"""
Storage layer — loads/saves LinUCB bandit matrices per user.

Matrices shape:
  A : (N_ACTIONS, N_FEATURES, N_FEATURES)  — covariance matrices
  b : (N_ACTIONS, N_FEATURES)              — reward-weighted feature vectors

N_FEATURES = 14  (5D affect one-hot + 9D personality vector)
N_ACTIONS  = 19  (DO_NOTHING + 9×INCREASE + 9×DECREASE)

Shape-mismatch guard: if saved matrices have the old shape (7, 7) they are
automatically re-initialised so the migration from v1 → v2 is seamless.
"""

from __future__ import annotations
import fcntl
import os
import re
import threading
from contextlib import contextmanager
from typing import Generator, Tuple

import numpy as np

N_ACTIONS  = 19
N_FEATURES = 14   # 5 affect + 9 personality

B_CLIP_MIN = -2.0
B_CLIP_MAX =  2.0

TABLE_DIR = os.environ.get("BANDIT_TABLE_DIR", "tables")

_thread_lock = threading.Lock()


def _sanitise_user_id(user_id: str) -> str:
    safe = re.sub(r"[^\w\-]", "_", user_id)
    return safe if safe else "default"


def _paths(user_id: str) -> Tuple[str, str, str]:
    uid  = _sanitise_user_id(user_id)
    base = os.path.join(TABLE_DIR, uid)
    return (
        f"{base}_bandit_A.npy",
        f"{base}_bandit_b.npy",
        f"{base}_bandit.lock",
    )


@contextmanager
def tables_locked(
    user_id: str = "default",
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    os.makedirs(TABLE_DIR, exist_ok=True)
    a_path, b_path, lock_path = _paths(user_id)

    with _thread_lock:
        lock_file = open(lock_path, "w")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            A, b = _load(a_path, b_path)
            yield A, b
            _save(A, b, a_path, b_path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()


def _load(a_path: str, b_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if os.path.exists(a_path) and os.path.exists(b_path):
        A = np.load(a_path)
        b = np.load(b_path)

        expected_A = (N_ACTIONS, N_FEATURES, N_FEATURES)
        expected_b = (N_ACTIONS, N_FEATURES)

        if A.shape != expected_A or b.shape != expected_b:
            print(
                f"[storage] Matrix shape {A.shape}/{b.shape} != "
                f"expected {expected_A}/{expected_b}. Re-initialising."
            )
            return _init_matrices()

        b = np.clip(b, B_CLIP_MIN, B_CLIP_MAX)
        return A, b

    return _init_matrices()


def _save(A: np.ndarray, b: np.ndarray, a_path: str, b_path: str) -> None:
    os.makedirs(TABLE_DIR, exist_ok=True)
    np.save(a_path, A)
    np.save(b_path, b)


def _init_matrices() -> Tuple[np.ndarray, np.ndarray]:
    A = np.array([np.eye(N_FEATURES) for _ in range(N_ACTIONS)])
    b = np.zeros((N_ACTIONS, N_FEATURES))
    return A, b


def load_tables(user_id: str = "default") -> Tuple[np.ndarray, np.ndarray]:
    os.makedirs(TABLE_DIR, exist_ok=True)
    a_path, b_path, _ = _paths(user_id)
    return _load(a_path, b_path)


def save_tables(A: np.ndarray, b: np.ndarray, user_id: str = "default") -> None:
    a_path, b_path, _ = _paths(user_id)
    _save(A, b, a_path, b_path)


def reset_tables(user_id: str = "default") -> None:
    A, b = _init_matrices()
    a_path, b_path, _ = _paths(user_id)
    _save(A, b, a_path, b_path)
