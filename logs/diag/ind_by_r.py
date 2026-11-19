"""Diagnostic only (not used for any choice): in-distribution accuracy of saved e1 square models at r = 4, 5, 6."""
import glob, re, sys, types
import numpy as np, torch
sys.path.insert(0, "/home/minjun_dev/Introduction_LLM/Square_Attention")
import poc_fair as P
torch.set_num_threads(4)
args = types.SimpleNamespace(D_train=4, n=16, m=4)
print("file | taps key/value/read (softmax at scale 10) | in-dist at r=4 | r=5 | r=6")
for f in sorted(glob.glob("/home/minjun_dev/Introduction_LLM/Square_Attention/ckpt/e1_square_lr1e-3_s*.pt")):
    seed = int(re.search(r"_s(\d+)\.pt", f).group(1))
    m = P.Tied(16, 4, read="square", gate_ctx=True)
    m.load_state_dict(torch.load(f, map_location="cpu")); m.tap_scale = 10.0
    taps = torch.softmax(m.taps * 10, 1).detach().numpy().round(2).tolist()
    accs = []
    for r in (4, 5, 6):
        ind, _, _ = P.evaluate(m, types.SimpleNamespace(**vars(args)), np.random.default_rng(10_000 + seed), "cpu", r, n_eval=1000) if False else (None, None, None)
        rng = np.random.default_rng(10_000 + seed)
        res = {}
        with torch.no_grad():
            for D in range(1, 5):
                tok, lab, qd, _ = P.make_batch(rng, 1000, 16, 4, 4, q0_depth=D)
                pred = m(torch.from_numpy(tok), r=r)[:, 0].argmax(-1).numpy()
                ok = qd[:, 0] == D
                res[D] = round(float((pred[ok] == lab[ok, 0]).mean()), 3)
        accs.append(list(res.values()))
    print(f"s{seed} | {taps} | {accs[0]} | {accs[1]} | {accs[2]}", flush=True)
