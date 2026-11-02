"""Train network.py's network on the degraded Metrica games (torch; training only).

    uv run --group train more-than-a-hack-2027 train-seq            # the shipped models/seq_v1.json

Data: every degradation seed in `seeds` of every game in `games`, with each copy also seen with
the teams swapped (network.swap). Labels are metrica.py's event-derived H/A/D on graded frames.
Batches are random 160 s crops of a half.

Determinism: fixed torch / numpy seeds, deterministic algorithms and a fixed thread count, so the
same command on the same machine rewrites the same JSON byte for byte.
"""

import hashlib
import os

import numpy as np

from .. import _paths
from . import network

TRAIN_DEFAULTS = {
    "channels": 64,
    "stride": 5,
    "dilations": (1, 2, 4, 8, 16, 32),
    "dropout": 0.3,
    "steps": 800,
    "batch": 12,
    "crop": 4000,
    "lr": 2e-3,
    "weight_decay": 0.05,
    "input_noise": 0.1,
    "seed": 0,
    "threads": 6,
    "short_weight": 0.0,    # extra loss weight on frames of possessions shorter than short_s
    "short_s": 4.0,
    "turn_weight": 0.0,     # extra loss weight within turn_frames of a possession change
    "turn_frames": 12,
    "flip": False,          # also train on left-right mirrored copies, and average them at test
    "aux": (),              # cached input files (CACHE_DIR) of an auxiliary dataset, e.g. the
                            # open DFL matches
    "aux_p": 0.5,           # share of batch rows drawn from the auxiliary dataset
    "aux_until": 1.0,       # share of the steps that use it; the rest fine-tune on the main data
    "aux_head": "own",      # "own": learned through a second output head; "main": the aux labels
                            # are the same kind as the main ones, so they train the main head
}
CACHE_DIR = os.path.join(_paths.CACHE_DIR, "seq_inputs")


def game_inputs(seed, game):
    """(X, swap_sign, labels, graded, halves) for one degraded Metrica game, cached on disk."""
    from ..validation import harness
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "s{}_g{}.npz".format(seed, game))
    if os.path.exists(path):
        z = np.load(path)
        return (z["X"], z["swap_sign"], z["labels"], z["graded"],
                tuple(tuple(int(v) for v in row) for row in z["halves"]))
    loaded = harness.load_games((game,), ("degraded",), degrade_cfg={"seed": seed})
    eg = loaded[(game, "degraded")]
    X, swap_sign = network.inputs(eg.ctx)
    graded = eg.ctx.half_of_frame > 0
    np.savez(path, X=X, swap_sign=swap_sign, labels=eg.labels, graded=graded,
             halves=np.array(eg.ctx.halves))
    return X, swap_sign, eg.labels, graded, eg.ctx.halves


def load_cached(name):
    z = np.load(os.path.join(CACHE_DIR, name))
    return (z["X"], z["swap_sign"], z["labels"], z["graded"],
            tuple(tuple(int(v) for v in row) for row in z["halves"]))


def _net(cin, cfg, heads=1):
    import torch.nn as nn
    import torch.nn.functional as F

    class LayerNormC(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.weight = nn.Parameter(__import__("torch").ones(ch))
            self.bias = nn.Parameter(__import__("torch").zeros(ch))

        def forward(self, x):
            return F.layer_norm(x.transpose(1, 2), (x.shape[1],), self.weight,
                                self.bias).transpose(1, 2)

    class Block(nn.Module):
        def __init__(self, ch, d, drop):
            super().__init__()
            self.n = LayerNormC(ch)
            self.c1 = nn.Conv1d(ch, ch, 3, padding=d, dilation=d)
            self.c2 = nn.Conv1d(ch, ch, 1)
            self.drop = nn.Dropout(drop)

        def forward(self, x):
            z = self.c1(F.gelu(self.n(x), approximate="tanh"))
            return x + self.c2(self.drop(F.gelu(z, approximate="tanh")))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            ch, s = cfg["channels"], cfg["stride"]
            self.down = nn.Conv1d(cin, ch, s, stride=s)
            self.blocks = nn.ModuleList([Block(ch, d, cfg["dropout"]) for d in cfg["dilations"]])
            self.up = nn.ConvTranspose1d(ch, ch, s, stride=s)
            self.skip = nn.Conv1d(cin, ch, 5, padding=2)
            self.head = nn.Sequential(nn.GELU(approximate="tanh"), nn.Conv1d(ch, ch, 5, padding=2),
                                      nn.GELU(approximate="tanh"), nn.Conv1d(ch, 3 * heads, 1))

        def forward(self, x):
            # pad to a whole number of strides, as network.SeqModel.forward does
            n = x.shape[-1]
            x = F.pad(x, (0, (-n) % self.down.stride[0]))
            h = self.down(x)
            for block in self.blocks:
                h = block(h)
            return self.head(self.up(h) + self.skip(x))[..., :n]

    return Net()


def _weights(y, cfg, fps=25):
    """Per-frame loss weights for one half: 1, plus extras on short possessions and turnovers."""
    w = np.ones(len(y))
    if not (cfg["short_weight"] or cfg["turn_weight"]):
        return w
    change = np.flatnonzero(np.diff(y) != 0) + 1
    starts = np.r_[0, change]
    stops = np.r_[change, len(y)]
    short = int(cfg["short_s"] * fps)
    for a, b in zip(starts, stops):
        if y[a] in (0, 1) and b - a < short:
            w[a:b] += cfg["short_weight"]
    k = int(cfg["turn_frames"])
    for t in change:
        if y[t - 1] in (0, 1) and y[t] in (0, 1):
            w[max(0, t - k):t + k] += cfg["turn_weight"]
    return w


def _sequences(blocks, cfg):
    """[(X half, y half, w half, swap)] for every game copy and its team swap; y = -100 where
    unscored. The swapped copy shares X and is swapped crop by crop (swap = its swap_sign), which
    halves the memory and gives the same numbers."""
    seqs = []
    for X, swap_sign, labels, graded, halves in blocks:
        y = np.where(graded & (labels < 3), labels, -100).astype(np.int64)
        ys = y.copy()
        ys[y == 0], ys[y == 1] = 1, 0
        for _, first, last in halves:
            part = slice(first, last + 1)
            w = _weights(y[part], cfg)
            seqs.append((X[part], y[part], w, None))
            seqs.append((X[part], ys[part], w, swap_sign))
    return seqs


def _mirror(cfg, n_channels):
    if not cfg["flip"]:
        return None
    return network.flip_sign(n_channels).astype(np.float32)


def train(blocks, cfg=None, log=print, meta=None, device="cpu"):
    """Fit on `blocks` (game_inputs tuples). Returns a network.SeqModel.

    device: "cpu" (the shipped, byte-reproducible path) or a torch device such as "cuda" (e.g. on
    a Kaggle GPU, for experiments). Batches and noise are drawn on the CPU either way, so only
    the arithmetic moves; GPU results are repeatable on the same GPU but not equal to the CPU's."""
    import torch
    import torch.nn.functional as F

    cfg = dict(TRAIN_DEFAULTS, **(cfg or {}))
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(int(cfg["threads"]))
    torch.use_deterministic_algorithms(True, warn_only=device != "cpu")
    rng = np.random.RandomState(cfg["seed"])

    seqs = _sequences(blocks, cfg)
    aux = _sequences([load_cached(name) for name in cfg["aux"]], cfg) if cfg["aux"] else []
    if cfg["aux_head"] not in ("own", "main"):
        raise ValueError("aux_head must be 'own' or 'main'")
    heads = 2 if aux and cfg["aux_head"] == "own" else 1
    cin = seqs[0][0].shape[1]
    geo = network.BALL_BLOCK + 2 * network.TEAM_BLOCK
    scale = np.ones(cin)
    rest = np.concatenate([x[:, geo:] for x, _, _, _ in seqs[::2]]).astype(np.float64)
    scale[geo:] = np.sqrt((rest ** 2).mean(axis=0)) + 1e-3
    del rest
    scale32 = scale.astype(np.float32)

    net = _net(cin, cfg, heads).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["steps"],
                                                pct_start=0.1)
    pools = [seqs] + ([aux] if aux else [])
    probs = []
    for pool in pools:
        lengths = np.array([len(x) for x, _, _, _ in pool], dtype=np.float64)
        probs.append(lengths / lengths.sum())
    crop = int(cfg["crop"])
    mirror = _mirror(cfg, cin)
    net.train()
    for step in range(int(cfg["steps"])):
        xb, yb, wb, hb = [], [], [], []
        aux_p = cfg["aux_p"] if step < cfg["aux_until"] * cfg["steps"] else 0.0
        for _ in range(int(cfg["batch"])):
            k = int(bool(aux) and rng.rand() < aux_p)
            pool = pools[k]
            x, y, w, sign = pool[rng.choice(len(pool), p=probs[k])]
            hb.append(k if heads > 1 else 0)
            start = rng.randint(0, max(1, len(x) - crop))
            xs, ys, ws = x[start:start + crop], y[start:start + crop], w[start:start + crop]
            if sign is not None:
                xs = network.swap(xs, sign)
            if mirror is not None and rng.rand() < 0.5:
                xs = xs * mirror
            if len(xs) < crop:
                xs = np.pad(xs, ((0, crop - len(xs)), (0, 0)))
                ys = np.pad(ys, (0, crop - len(ys)), constant_values=-100)
                ws = np.pad(ws, (0, crop - len(ws)))
            xb.append((xs / scale32).T)
            yb.append(ys)
            wb.append(ws)
        xb = torch.from_numpy(np.stack(xb).astype(np.float32))
        yb = torch.from_numpy(np.stack(yb))
        wb = torch.from_numpy(np.stack(wb).astype(np.float32))
        if cfg["input_noise"] > 0:
            xb = xb + cfg["input_noise"] * torch.randn(xb.shape)
        xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
        logits = net(xb)
        if heads > 1:
            rows = torch.arange(len(hb), device=device)
            logits = logits.view(len(hb), heads, 3, -1)[rows, torch.tensor(hb, device=device)]
        per_frame = F.cross_entropy(logits, yb, ignore_index=-100, reduction="none")
        valid = (yb != -100).float() * wb
        loss = (per_frame * valid).sum() / valid.sum().clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if log and (step % 100 == 0 or step == cfg["steps"] - 1):
            log("step {} loss {:.4f}".format(step, loss.item()))

    weights = {k: v.detach().cpu().numpy().astype(np.float64)
               for k, v in net.state_dict().items()}
    arch = {"stride": cfg["stride"], "channels": cfg["channels"],
            "dilations": list(cfg["dilations"]), "flip_tta": bool(cfg["flip"]), "heads": heads}
    info = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
    info.update(meta or {})
    return network.SeqModel(arch, scale, weights, info)


def logo_models(seeds, games, cfg=None, log=print):
    """{held-out game: model trained on the other games (all `seeds`)}, cached under .cache/."""
    cfg = dict(TRAIN_DEFAULTS, **(cfg or {}))
    key = hashlib.sha256(repr((sorted(cfg.items()), tuple(seeds), tuple(games))).encode())
    tag = key.hexdigest()[:12]
    out = {}
    for g in games:
        path = os.path.join(CACHE_DIR, "logo_{}_g{}.json".format(tag, g))
        if os.path.exists(path):
            out[g] = network.SeqModel.from_json(path)
            continue
        blocks = [game_inputs(s, h) for s in seeds for h in games if h != g]
        if log:
            log("training held-out-{} model ({} game copies)".format(g, len(blocks)))
        model = train(blocks, cfg, log=log, meta={"held_out": g, "seeds": list(seeds)})
        os.makedirs(CACHE_DIR, exist_ok=True)
        model.to_json(path)
        out[g] = model
    return out
