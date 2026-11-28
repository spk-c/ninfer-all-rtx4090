"""Experimental CPU reference for Inkling2 rank-1 GGUF adapters.

This is a numerical oracle only, not a NInfer runtime adapter or an artifact
conversion. Quantized base rows remain unchanged on disk.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
from gguf import GGUFReader


class Rank1Adapter:
    def __init__(self, path: str | Path):
        reader = GGUFReader(str(path))
        fields = reader.fields
        for key, expected in {
            "general.architecture": "inkling2_loop",
            "general.type": "adapter",
            "adapter.type": "lora",
            "adapter.lora_alpha": 1.0,
        }.items():
            if key not in fields or fields[key].contents() != expected:
                raise ValueError(f"Unexpected adapter metadata: {key}")
        tensors = {t.name: t for t in reader.tensors}
        self.pairs = {}
        expected = {"token_embd.weight": ((201024, 1), (2304, 1))}
        for layer in range(24):
            prefix = f"blk.{layer}."
            expected[prefix + "attn_output.weight"] = ((1, 3072), (2304, 1))
            expected[prefix + "ffn_down_exps.weight"] = ((128, 1, 1152), (128, 2304, 1))
        if set(tensors) != {name + ".lora_" + side for name in expected for side in ("a", "b")}:
            raise ValueError("Adapter tensor set does not match Inkling2 rank-one layout")
        for name, (shape_a, shape_b) in expected.items():
            arrays = []
            for side, shape in (("a", shape_a), ("b", shape_b)):
                t = tensors[name + ".lora_" + side]
                if t.tensor_type.name != "F32" or tuple(t.shape[::-1]) != shape:
                    raise ValueError(f"Unsupported tensor layout: {t.name}")
                value = np.asarray(t.data, dtype=np.float64).reshape(shape)
                if not np.isfinite(value).all():
                    raise ValueError(f"Nonfinite tensor: {t.name}")
                arrays.append(value)
            self.pairs[name] = tuple(arrays)

    def delta_rows(self, name: str, begin: int, end: int) -> np.ndarray:
        a, b = self.pairs[name]
        if name == "token_embd.weight":
            return a[begin:end] @ b.T
        if a.ndim == 2:
            return b[begin:end] @ a
        # GGUF expert banks are flattened in expert-major, output-row order.
        idx = np.arange(begin, end)
        expert, row = np.divmod(idx, b.shape[1])
        return b[expert, row, :] * a[expert, 0, :]

    def apply_rows(self, name, begin, rows, scale):
        if not np.isfinite(scale):
            raise ValueError("Adapter scale must be finite")
        if scale == 0 or name not in self.pairs:
            return rows
        delta = self.delta_rows(name, begin, begin + rows.shape[0])
        if delta.shape != rows.shape:
            raise ValueError(f"Base/adapter shape mismatch: {name}")
        return rows + scale * delta


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
