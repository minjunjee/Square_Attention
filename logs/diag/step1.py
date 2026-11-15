"""Finds the configuration of a CPU trial log from its step-1 loss (no training, no eval)."""
import itertools
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/minjun_dev/poc_squared_read")
import poc_fair as P  # noqa: E402

torch.set_num_threads(1)


def step1(seed, gate_ctx, hard_taps, r_lo, r_hi, tap_sharp, steps=2500, ramp=0.5):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = P.Tied(16, 4, 64, 64, read="square", gate_ctx=bool(gate_ctx), hard_taps=hard_taps)
    tok, lab, _, _ = P.make_batch(rng, 64, 16, 4, 4)
    r = int(rng.integers(r_lo, r_hi + 1))
    m.tap_scale = 1.0 + (tap_sharp - 1.0) * min(1.0, 1 / (ramp * steps))
    with torch.no_grad():
        logits = m(torch.from_numpy(tok), r=r)
        return F.cross_entropy(logits.reshape(-1, P.V), torch.from_numpy(lab).reshape(-1)).item()


for seed, want in ((0, 3.6687), (1, 3.7828)):
    for g, ht, rh, ts in itertools.product((0, 1), (1, 2), (4, 5), (1, 10)):
        loss = step1(seed, g, ht, 2, rh, ts)
        if abs(loss - want) < 6e-5:
            print(f"seed {seed} gate_ctx {g} hard_taps {ht} r_hi {rh} tap_sharp {ts}: {loss:.4f}")
