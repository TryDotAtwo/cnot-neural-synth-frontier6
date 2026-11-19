"""Golden CPU results from the unchanged source beam driver and deterministic scorer callbacks."""
import argparse
from collections import deque
from hashlib import sha256
import importlib
import json
from pathlib import Path
import random
import sys

import torch

SOURCE = {
    "neural_beam": "37fcd8a24a8b202bd93f3b4520deb9c32655da0b67089ae01e5d96d0991f1bb3",
    "neighbors": "06b0ccac2f81f5f970697daf199824faa11d988379f90a8d89932434b7528969",
    "packed_walk": "270b2fa45b800f67050bcc4c66e2b9e21743e6aaa737eb390f5d92858ae9d153",
}


def load_beam(root):
    for module, digest in SOURCE.items():
        path = root / "research_v3" / "gpu" / (module + ".py")
        assert sha256(path.read_bytes()).hexdigest() == digest, path
    sys.path.insert(0, str(root))
    return importlib.import_module("research_v3.gpu.neural_beam").beam_search


def child(state, c, t):
    rows = list(state)
    rows[t] ^= rows[c]
    return tuple(rows)


def matrices(n):
    start = tuple(1 << i for i in range(n))
    gates = [(c, t) for c in range(n) for t in range(n) if c != t]
    if n <= 3:
        seen, queue = {start}, deque([start])
        while queue:
            s = queue.popleft()
            for c, t in gates:
                x = child(s, c, t)
                if x not in seen:
                    seen.add(x)
                    queue.append(x)
        assert len(seen) == {2: 6, 3: 168}[n]
        return sorted(seen)
    if n == 65:
        return [start, child(start, 63, 64), child(start, 64, 63),
                child(child(start, 63, 64), 0, 64)]
    rng = random.Random(94153)
    out = []
    for i in range(30):
        s = start
        for _ in range(1 + i % 14):
            s = child(s, *rng.choice(gates))
        out.append(s)
    s = start
    for action in (1, 4, 8, 10):
        s = child(s, *gates[action])
    out.append(s)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    beam = load_beam(args.source_root.resolve())
    lines = []
    outcomes = {}
    for n in (2, 3, 4, 65):
        action_count = n * (n - 1)
        for i, state in enumerate(matrices(n)):
            ties = i % 3 == 0
            width = (1, 4, 9, 5, 8)[i % 5]
            k = (1, min(3, action_count), action_count, action_count + 1, min(5, action_count))[i % 5]
            depth, limit = (i % 5) + 1, (1, 4, 25, 50)[i % 4]
            batch = 1 + (i % 7)
            if n == 65:
                width, k, depth, limit, batch = 2, 3, 2, 6, 3
            if n == 4 and i == 30:
                width, k, depth, limit, batch, ties = 7, 1, 9, 40, 5, True
            words = (n + 63) // 64
            packed = [(row >> (64 * j)) & ((1 << 64) - 1) for row in state for j in range(words)]
            signed = [word - (1 << 64) if word >= 1 << 63 else word for word in packed]
            root = torch.tensor(signed, dtype=torch.int64).reshape(1, n, words)
            counted = [0]

            def actions(rows):
                counted[0] += len(rows)
                q = torch.arange(action_count, dtype=torch.float32).remainder(3)
                return (q.zero_() if ties else q).expand(len(rows), -1)

            def values(rows):
                v = rows.sum(dim=(1, 2)).remainder(5).float()
                return v.zero_() if ties else v

            result = beam(root, actions, values, beam_width=width, shortlist=k, max_depth=depth,
                          max_expanded=limit, inference_batch=batch)
            if result.word is not None:
                assert result.verified
            budget = dict(beam_width=width, shortlist=k, max_depth=depth, max_expanded=limit, inference_batch=batch)
            row = dict(n=n, packed=packed, ties=ties, budget=budget, status=result.status, word=result.word,
                       expanded=result.expanded, generated=result.generated, values=result.value_evaluations,
                       actions=counted[0])
            lines.append(json.dumps(row, separators=(",", ":")))
            outcomes[result.status] = 1 + outcomes.get(result.status, 0)
    payload = ("\n".join(lines) + "\n").encode()
    args.output.write_bytes(payload)
    print("torch", torch.__version__, "reference", SOURCE["neural_beam"], "cases", len(lines),
          "outcomes", outcomes, "sha256", sha256(payload).hexdigest())


if __name__ == "__main__":
    main()
