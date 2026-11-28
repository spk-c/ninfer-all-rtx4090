from pathlib import Path
import argparse
import numpy as np
from tools.reference.inkling2_rank1_adapter import Rank1Adapter, sha256
from tools.reference.inkling2 import Weights
from unittest.mock import patch

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--adapter", type=Path, default=root / "atomicchat-adapter" / "Inkling-2-Mini-Uncensored-LoRA.gguf")
parser.add_argument("--base", type=Path, help="Optional real GSQ-RCO GGUF for row validation")
args = parser.parse_args()
path = args.adapter
adapter = Rank1Adapter(path)
rng = np.random.default_rng(42)
count = 0
for name, (a, b) in adapter.pairs.items():
    if name == "token_embd.weight":
        start, stop = 121, 125
        expected = np.outer(a[start:stop,0], b[:,0])
    elif a.ndim == 2:
        start, stop = 17, 21
        expected = np.outer(b[start:stop,0], a[0])
    else:
        start, stop = 2302, 2306  # crosses expert 0 -> 1
        expected = np.stack([b[i//2304, i%2304,0] * a[i//2304,0] for i in range(start,stop)])
    base = rng.normal(size=expected.shape)
    actual = adapter.delta_rows(name,start,stop)
    np.testing.assert_array_equal(actual,expected)
    original = base.copy()
    for scale in (0., 0.5, 1.):
        np.testing.assert_allclose(adapter.apply_rows(name,start,base,scale),base+scale*expected,rtol=0,atol=0)
        weights = Weights(None, adapter, scale)
        with patch.object(weights, "_base_rows", return_value=base):
            np.testing.assert_allclose(weights.rows(name,start,stop),base+scale*expected,rtol=0,atol=0)
        np.testing.assert_array_equal(base,original)
        count += 1
    assert adapter.apply_rows(name,start,base,0.) is base
base = np.zeros((2,3))
assert adapter.apply_rows("untouched.weight",0,base,1.) is base
print("PASS: actual adapter", len(adapter.pairs), "rank-one pairs;",count,"row-update cases")
print("SHA256:",sha256(path))

if args.base:
    from tools.convert.sources.gguf import GGUFFile
    real_count = 0
    with GGUFFile(args.base) as gguf:
        plain = Weights(gguf)
        adapted = Weights(gguf, adapter, 1.0)
        disabled = Weights(gguf, adapter, 0.0)
        for name,(a,b) in adapter.pairs.items():
            shape = tuple(gguf.info(name).shape)
            if name == "token_embd.weight":
                expected_shape = (a.shape[0], b.shape[0])
                begin,end = 121,125
            elif a.ndim == 2:
                expected_shape = (b.shape[0],a.shape[1])
                begin,end = 17,21
            else:
                expected_shape = (a.shape[0],b.shape[1],a.shape[2])
                begin,end = 2302,2306
            assert shape == expected_shape, (name,shape,expected_shape)
            w = plain.rows(name,begin,end)
            z = adapted.rows(name,begin,end)
            np.testing.assert_array_equal(disabled.rows(name,begin,end),w)
            np.testing.assert_allclose(z,w+adapter.delta_rows(name,begin,end),rtol=0,atol=0)
            assert np.isfinite(z).all(), name
            real_count += 1
    print("PASS:",real_count,"real GSQ-RCO base tensor shape and row-update checks")
