"""Train sequence models (v3) on a Kaggle GPU: held-out runs for evaluation, or the shipped ones.

    uv run python scripts/kaggle_seq.py v3-final --push      # the shipped ensemble (models/seq_v3_*)
    uv run python scripts/kaggle_seq.py v3-heldout-01 --push # held-out runs of members s0 and s1
    uv run python scripts/kaggle_seq.py dbv-atd-01 --push    # the dead-ball view experiment

v3-MODE[-MEMBERS] trains the v3 recipe's members (digits: training seeds, "o": the own-head
variant; all eight when omitted) in mode "final" or "heldout".
    uv run python scripts/kaggle_seq.py NAME JOBS_JSON [--push] [--cpu]

Why Kaggle: training one network takes ~5 min on a T4 and far longer on a laptop CPU. Only the
weights come back; predicting needs numpy alone (seq/network.py).

What is uploaded. The kernel embeds this checkout's _paths.py, seq/network.py and seq/training.py
(no organiser code: data/context.py becomes a two-constant stub) and reads private Kaggle datasets
of cached sequence-model inputs, all built from OPEN data only, never from data/ (the competition
data may not be redistributed):

    mtah-seq-inputs      Metrica games 1-3, degradation seeds 0-5 (s{seed}_g{game}.npz), and DFL
                         control-labelled matches, seeds 0-1 (dflc_*)
    mtah-seq-dflc-extra  DFL seeds 2-5
    mtah-seq-scc         SkillCorner, one-touch labels, seeds 0-2 (scc_*)
    mtah-seq-touch       the touch channels of all of the above (*_t.npz, key T), joined to the
                         base files at load time
    mtah-seq-view        DFL (seeds 0-5) and SkillCorner (seeds 0-2) with the ball shown while
                         dead as in the competition ATD's stoppages (dflcv_*_t, sccv_*_t; full
                         inputs, touch included)

They are the files `train-seq`, `dfl-inputs --labels control` and `skillcorner-inputs --labels
control` (each with and without --touch) write to .cache/seq_inputs/, compressed. The degraded
copies replay BAR-ATH's per-frame missing-player counts (a list of integers, no tracking).

Jobs: [{"tag", "cfg" (training.train settings), "touch" (bool), "mode" ("heldout" | "final"),
optional "aux_by_game" ({game: aux files}, replacing cfg["aux"] for that held-out game) and
"eval_by_game" ({game: more input files to predict with that game's network})}].
held-out: for each Metrica game, train on the other two (seeds 0-5) and write {tag}_g{game}.json
and lp_{tag}.npz, the held-out log-probabilities on evaluation seeds 0-2 (keys s{seed}_g{game}),
which `tune-seq --logprobs` scores, and lp_{tag}_extra.npz for eval_by_game (keys: the files'
base names). final: train on all three games and write {tag}.json.
Outputs: `kaggle kernels output <user>/mtah-seq-NAME`. GPU training uses torch's deterministic
settings where they exist (warn_only: a CUDA op without a deterministic version only warns), so a
rerun can differ in the last bits, and it is not bit-equal to CPU training. The shipped weights are
therefore the record; scripts/check_reproducibility.sh rebuilds the submission from them.
"""
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, "src", "more_than_a_hack_2027")
USER = os.environ.get("MTAH_KAGGLE_USER", "mertkamberov")
DATASETS = [USER + "/mtah-seq-inputs", USER + "/mtah-seq-dflc-extra", USER + "/mtah-seq-scc",
            USER + "/mtah-seq-touch", USER + "/mtah-seq-view"]

# The shipped v3 recipe (docs/TUNING_V3.md): the Metrica games plus the open DFL (control labels,
# 6 degradation seeds) and SkillCorner (one-touch labels, 3 seeds) matches on the main head, half
# of every batch, with the touch inputs; seven training seeds and one auxiliary-head variant.
# V3_VIEW is how those matches show the ball while it is dead: "atd" replays the competition ATD's
# stoppages (external/degrade.py, dead_ball_view), as chosen by the dead-ball view experiment
# below; "edges" hides it except at the edges, the recipe frozen on 2026-11-02.
V3_RECIPE = {"short_weight": 2.0, "turn_weight": 1.0, "steps": 3200, "aux_head": "main"}
V3_SEEDS = (0, 1, 2, 3, 4, 5, 6)
V3_VIEW = "atd"


def v3_jobs(mode="final", members=None, view=V3_VIEW):
    sys.path.insert(0, os.path.join(REPO, "src"))
    from more_than_a_hack_2027.external import dfl, skillcorner
    aux = [dfl.input_file(m, s, "control", True, view) for m in dfl.MATCHES for s in range(6)]
    aux += [skillcorner.input_file(m, s, "control", True, view)
            for m in skillcorner.match_ids() for s in range(3)]
    cfg = dict(V3_RECIPE, aux=aux)
    jobs = [{"tag": "seq_v3_s{}".format(s), "touch": True, "mode": mode,
             "cfg": dict(cfg, seed=s)} for s in V3_SEEDS]
    jobs.append({"tag": "seq_v3_own", "touch": True, "mode": mode,
                 "cfg": dict(cfg, aux_head="own")})
    if members is not None:
        keep = {"seq_v3_" + ("own" if c == "o" else "s" + c) for c in members}
        jobs = [j for j in jobs if j["tag"] in keep]
        if len(jobs) != len(keep):
            raise SystemExit("unknown v3 members: " + members)
    return jobs


# The dead-ball view experiment (docs/PLAN.md, "Sequence model (v3)"): the v3 recipe trained with the
# DFL and SkillCorner inputs as before ("edges") or with the ball shown while dead as in the ATD
# ("atd"). Each held-out Metrica game also holds out a group of DFL matches, so both arms are
# scored on DFL matches they never saw, with the ball shown as in the ATD (dflcv) and as before.
DFL_FOLDS = {1: (0, 1, 2), 2: (3, 4), 3: (5, 6)}       # indices into dfl.MATCHES


def view_jobs(arm, seeds):
    sys.path.insert(0, os.path.join(REPO, "src"))
    from more_than_a_hack_2027.external import dfl, skillcorner
    aux_by_game, eval_by_game = {}, {}
    for g, held in DFL_FOLDS.items():
        train_dfl = [m for i, m in enumerate(dfl.MATCHES) if i not in held]
        test_dfl = [dfl.MATCHES[i] for i in held]
        aux_by_game[g] = ([dfl.input_file(m, s, "control", True, arm)
                           for m in train_dfl for s in range(6)]
                          + [skillcorner.input_file(m, s, "control", True, arm)
                             for m in skillcorner.match_ids() for s in range(3)])
        eval_by_game[g] = [dfl.input_file(m, s, "control", True, view)
                           for view in ("atd", "edges") for m in test_dfl for s in range(3)]
    return [{"tag": "dbv_{}_s{}".format(arm, s), "touch": True, "mode": "heldout",
             "cfg": dict(V3_RECIPE, seed=s), "aux_by_game": aux_by_game,
             "eval_by_game": eval_by_game} for s in seeds]


KERNEL = r'''
import glob, json, os, sys, time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
SOURCES = __SOURCES__
JOBS = __JOBS__
DEVICE = __DEVICE__
WORK = os.environ.get("MTAH_WORK", "/kaggle/working")
INPUT = os.environ.get("MTAH_INPUT", "/kaggle/input")
root = os.path.join(WORK, "pkg/more_than_a_hack_2027")
os.makedirs(root, exist_ok=True)
for name, text in SOURCES.items():
    os.makedirs(os.path.dirname(os.path.join(root, name)), exist_ok=True)
    with open(os.path.join(root, name), "w") as f:
        f.write(text)
sys.path.insert(0, os.path.join(WORK, "pkg"))
import torch
from more_than_a_hack_2027.seq import network, training
if DEVICE == "cuda" and not torch.cuda.is_available():
    raise SystemExit("no GPU in this session")
print("torch", torch.__version__, "device", DEVICE,
      torch.cuda.get_device_name(0) if DEVICE == "cuda" else "", flush=True)
INDEX = {}                                   # input files of every attached dataset, by name
for path in sorted(glob.glob(os.path.join(INPUT, "**", "*.npz"), recursive=True)):
    INDEX.setdefault(os.path.basename(path), path)


def load_by_name(name):
    z = np.load(INDEX[name])
    if "X" not in z.files:                   # a touch-only file (mtah-seq-touch): join to its base
        T = z["T"]
        z = np.load(INDEX[name[:-len("_t.npz")] + ".npz"])
        X = np.hstack([z["X"], T]).astype(np.float32)
        sign = np.concatenate([z["swap_sign"], np.ones(T.shape[1])])
        assert X.shape[1] == network.N_BASE + 2 * network.TOUCH_BLOCK
    else:
        X, sign = z["X"], z["swap_sign"]
    return (X, sign, z["labels"], z["graded"],
            tuple(tuple(int(v) for v in row) for row in z["halves"]))


training.load_cached = load_by_name          # train() loads its "aux" files through this
print(len(INDEX), "input files", flush=True)
BLOCKS = {}


def metrica_blocks(touch):
    if touch not in BLOCKS:
        BLOCKS[touch] = {(s, g): training.load_cached(training.input_name("s{}_g{}".format(s, g), touch))
                         for s in range(6) for g in (1, 2, 3)}
    return BLOCKS[touch]


def gpu_logprobs_halves(model, X, sign, halves):
    """network.logprobs_halves on the GPU: same graph (training._net), float64, same averaging."""
    arch = model.arch
    cfg = {"channels": arch["channels"], "stride": arch["stride"], "dilations": arch["dilations"],
           "dropout": 0.0}
    net = training._net(X.shape[1], cfg, int(arch.get("heads", 1))).double().to(DEVICE).eval()
    net.load_state_dict({k: torch.from_numpy(np.asarray(v, dtype=np.float64))
                         for k, v in model.weights.items()})
    stride = int(arch["stride"])
    Xs = network.swap(X, sign)
    views = [(X, False), (Xs, True)]
    if arch.get("flip_tta"):
        mirror = network.flip_sign(X.shape[1]).astype(X.dtype)
        views += [(X * mirror, False), (Xs * mirror, True)]
    out = np.zeros((len(X), 3))
    with torch.no_grad():
        for _, first, last in halves:
            part = slice(first, last + 1)
            preds = []
            for V, swapped in views:
                x = (V[part] / model.scale).astype(np.float64)
                T = len(x)
                x = np.pad(x, ((0, (-T) % stride), (0, 0)))
                logits = net(torch.from_numpy(x.T[None]).to(DEVICE))[0, :3, :T]
                p = torch.log_softmax(logits, dim=0).T.cpu().numpy()
                preds.append(p[:, [1, 0, 2]] if swapped else p)
            mean = np.mean(preds, axis=0)
            m = mean.max(axis=1, keepdims=True)
            out[part] = mean - m - np.log(np.exp(mean - m).sum(axis=1, keepdims=True))
    return out


checked = False
for job in JOBS:
    tag, cfg = job["tag"], job["cfg"]
    blocks = metrica_blocks(bool(job.get("touch")))
    if job.get("mode") == "final":           # all three games: a shipped model, no held-out scores
        t = time.time()
        print("== {} final".format(tag), flush=True)
        model = training.train(list(blocks.values()), cfg, log=lambda m: print(m, flush=True),
                               meta={"games": [1, 2, 3], "seeds": list(range(6)), "device": DEVICE},
                               device=DEVICE)
        model.to_json(os.path.join(WORK, "{}.json".format(tag)))
        print("{} final done in {:.0f} s".format(tag, time.time() - t), flush=True)
        continue
    checked = False
    lps, extra = {}, {}
    for g in (1, 2, 3):
        t = time.time()
        train = [blocks[(s, h)] for s in range(6) for h in (1, 2, 3) if h != g]
        by_game = job.get("aux_by_game") or {}
        cfg_g = dict(cfg, aux=by_game.get(g, by_game.get(str(g)))) if by_game else cfg
        print("== {} held-out {}".format(tag, g), flush=True)
        model = training.train(train, cfg_g, log=lambda m: print(m, flush=True),
                               meta={"held_out": g, "seeds": list(range(6)), "device": DEVICE},
                               device=DEVICE)
        model.to_json(os.path.join(WORK, "{}_g{}.json".format(tag, g)))
        if not checked:                      # GPU inference == numpy inference, on a slice
            X, sign, _, _, _ = blocks[(0, g)]
            ref = network.logprobs_halves(model, X[:5000], sign, ((1, 0, 4999),))
            got = gpu_logprobs_halves(model, X[:5000], sign, ((1, 0, 4999),))
            print("gpu vs numpy max |diff|", float(np.abs(ref - got).max()), flush=True)
            assert np.abs(ref - got).max() < 1e-6
            checked = True
        for s in (0, 1, 2):
            X, sign, _, _, halves = blocks[(s, g)]
            lps["s{}_g{}".format(s, g)] = gpu_logprobs_halves(model, X, sign, halves).astype(np.float32)
        by_game = job.get("eval_by_game") or {}
        for name in by_game.get(g, by_game.get(str(g), [])):
            X, sign, _, _, halves = training.load_cached(name)
            extra[name[:-len(".npz")]] = gpu_logprobs_halves(model, X, sign, halves).astype(np.float32)
        print("{} held-out {} done in {:.0f} s".format(tag, g, time.time() - t), flush=True)
    np.savez(os.path.join(WORK, "lp_{}.npz".format(tag)), **lps)
    if extra:
        np.savez(os.path.join(WORK, "lp_{}_extra.npz".format(tag)), **extra)
print("ALL DONE", flush=True)
'''


def build(name, jobs, device="cuda"):
    # our own modules only; data/context.py (which imports the organiser's games.py) is a stub
    sources = {"__init__.py": "", "data/__init__.py": "", "seq/__init__.py": "",
               "data/context.py": "HOME, AWAY = 0, 1\n"}
    for f in ("_paths.py", "seq/network.py", "seq/training.py"):
        with open(os.path.join(PKG, f)) as handle:
            sources[f] = handle.read()
    out = os.path.join(REPO, ".cache", "kaggle", "kernel-" + name)
    os.makedirs(out, exist_ok=True)
    code = (KERNEL.replace("__SOURCES__", repr(sources)).replace("__JOBS__", repr(jobs))
            .replace("__DEVICE__", repr(device)))
    with open(os.path.join(out, "kernel.py"), "w") as handle:
        handle.write(code)
    slug = "mtah-seq-" + name
    meta = {"id": USER + "/" + slug, "title": slug, "code_file": "kernel.py", "language": "python",
            "kernel_type": "script", "is_private": "true",
            "enable_gpu": str(device == "cuda").lower(), "enable_tpu": "false",
            "enable_internet": "false", "dataset_sources": DATASETS, "competition_sources": [],
            "kernel_sources": [], "model_sources": []}
    with open(os.path.join(out, "kernel-metadata.json"), "w") as handle:
        json.dump(meta, handle, indent=1)
    return out


def main(argv):
    name = argv[0]
    if name.startswith("v3-"):                     # v3-<final|heldout>[-<members>], e.g. v3-heldout-01
        mode, _, members = name[len("v3-"):].partition("-")
        jobs = v3_jobs(mode, members or None)
    elif name.startswith("dbv-"):                  # dbv-<edges|atd>-<seed digits>, e.g. dbv-atd-01
        _, arm, digits = name.split("-")
        jobs = view_jobs(arm, [int(c) for c in digits])
    else:
        jobs = json.loads(argv[1])
    device = "cpu" if "--cpu" in argv else "cuda"
    out = build(name, jobs, device)
    print(out)
    if "--push" in argv:
        cmd = ["uvx", "--with", "six", "kaggle", "kernels", "push", "-p", out]
        if device == "cuda":
            cmd += ["--accelerator", "NvidiaTeslaT4"]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main(sys.argv[1:])
