"""Emit exact NumPy 2.2.6 shared CNOT words and complete packed left transforms."""
import hashlib
from pathlib import Path
import struct

import numpy as np


def main():
    if np.__version__ != "2.2.6":
        raise RuntimeError("this fixture requires NumPy 2.2.6")
    cases = [
        (1, 5, 1e100, 0), (2, 11, 0.0, 0), (2, 13, 4.9, 91),
        (3, 33, 10.0 / 3, 2**64 - 1), (4, 129, 3.75, 0x1234567800000000),
        (65, 13, 1.0, 91), (144, 7, 0.75, 2**64 - 1),
    ]
    data = bytearray(b"CNOTMC01" + struct.pack("<I", len(cases)))
    for n, samples, tau, seed in cases:
        rng = np.random.default_rng(seed)
        lengths, gates, transforms = [], [], []
        for _ in range(samples):
            rows = [1 << i for i in range(n)]
            steps = int(rng.poisson(n * tau)) if n > 1 else 0
            lengths.append(steps)
            for _ in range(steps):
                control = int(rng.integers(n))
                target = int(rng.integers(n - 1))
                target += target >= control
                gates.append(control * n + target)
                rows[target] ^= rows[control]
            for word in range((n + 63) // 64):
                transforms.extend((row >> (64 * word)) & (2**64-1) for row in rows)
        data.extend(struct.pack("<IIdQQ", n, samples, tau, seed, len(gates)))
        data.extend(np.asarray(lengths, dtype="<u8").tobytes())
        data.extend(np.asarray(gates, dtype="<u4").tobytes())
        data.extend(np.asarray(transforms, dtype="<u8").tobytes())
    path = Path(__file__).with_name("shared_word_numpy_2_2_6.bin")
    path.write_bytes(data)
    print(len(data), hashlib.sha256(data).hexdigest(), path.name)


if __name__ == "__main__":
    main()
