# MTAH 2027 Phase 1: ball possession, team plan

## Status

| milestone | state |
|---|---|
| **M0 platform** (P1) | **done** — data layer + cache, `Context`, CLI, Viterbi decoder, bit-exact baseline port, 14 tests passing |
| **Dead ball** (P2) | **v1 done.** `stages/deadball.py`, `stages/hmm2.py`, `stages/restart.py`, `stages/ball.py` (cleaning), `data/tagging.py`, pipeline method `v1`. Two robustness bugs found through P4 and fixed (see "What v1 implements"). Thresholds are still untuned priors |
| **Ball & control** (P3) | **control v1 done.** `stages/control.py`: a team-symmetric logistic model trained on the external games, shipped as `models/control_v1.json` (retrained Nov 1 on the corrected labels); pipeline method **`v2`** (current best). Held-out external spellF1 **0.669** since the external games also lose the ball in live play as BAR-ATH does (0.720 without those gaps; v1 0.434). Next: dead-ball calls that survive ball gaps, then more temporal context for the team model (see "Live-play ball gaps") |
| **External validation** (P4) | **harness v1 done; labels corrected Nov 1.** `external/metrica.py`, `external/degrade.py`, `validation/harness.py`, CLI `eval-external`; leave-one-game-out training for learned stages. Two label rules (`metrica.LABEL_RULES`): a BALL LOST never starts a possession, and a one-touch recovery is not one. Spells per game ~375 → ~245; most old "short possessions" were artefacts. `validation/tune.py` (CLI `tune` → [`docs/TUNING.md`](TUNING.md)) chooses decoder settings over 3 seeds under both label readings. Since Nov 1 `external/degrade.py` also hides the ball in live play at BAR-ATH's measured rate (see "Live-play ball gaps"). Next: search over P2 thresholds |
| Decoder (P1) | `decode.viterbi_min_dead`: D runs ≥ `min_dead_s` (1 s), which closed a shortcut that capped the switch penalty |
| **Sequence model** (v3) | **done and frozen (Nov 2).** `seq/network.py`, `seq/training.py`, `seq/tuning.py`: a temporal CNN over the per-frame features, trained on the Metrica games plus open DFL and SkillCorner matches on a Kaggle GPU (`scripts/kaggle_seq.py`), shipped as an ensemble of 8 networks (`models/seq_v3_*.json`). Held-out external spellF1 **0.845** (v2 0.669). See "Sequence model (v3)" |
| **M3 freeze tooling** | **done (Nov 1)**, ahead of the M2 work. `predict` defaults to the frozen method (`pipeline.SUBMISSION_METHOD`, **`"v3"` since Nov 2**, `"v2"` before); `ablate` writes the held-out ablation table [`docs/ABLATION.md`](ABLATION.md) (3 seeds, paired bootstrap). Re-frozen twice the same day: after the label fix (switch 25 → 50, model retrained; sha256 `46fa8c61…`), then after adding live-play ball gaps to the degradation (w_control 3.2, switch 70, model retrained): sha256 `f1d4a191…`, 260 possessions. See "Freeze and reproducibility (M3)" |
| Ball-free model + stretch (P5) | not started |

How to run what exists now:

```bash
uv sync
uv run more-than-a-hack-2027 cache      # parse the ATD once into .cache/ (~2 s)
uv run more-than-a-hack-2027 info       # sanity summary of the loaded match
uv run more-than-a-hack-2027 predict --method baseline --window 50 --out evaluation/results/x.csv
uv run more-than-a-hack-2027 deadball --table .cache/stoppages.csv   # dead-ball diagnostics
uv run more-than-a-hack-2027 predict --method v1 --out .cache/v1.csv  # dead ball + placeholder team evidence
uv run more-than-a-hack-2027 predict --out .cache/v3.csv              # the frozen submission method (v3)
uv run more-than-a-hack-2027 predict --method v2 --out .cache/v2.csv  # the previous submission method
uv run more-than-a-hack-2027 train-control              # refit models/control_v1.json (deterministic)
uv run more-than-a-hack-2027 eval-external              # Metrica games, clean + degraded (downloads once, ~15 s)
uv run more-than-a-hack-2027 eval-external --describe   # also the label and input statistics per game
uv run more-than-a-hack-2027 tune --jobs 3 --out docs/TUNING.md   # choose w_control / switch, 3 seeds (~5 min)
uv run more-than-a-hack-2027 ablate --out docs/ABLATION.md   # held-out ablation table, 3 seeds (~5 min)
uv run python scripts/kaggle_seq.py v3-heldout --push   # held-out v3 networks on a Kaggle GPU (~2 h)
uv run python scripts/kaggle_seq.py v3-final --push     # the shipped v3 ensemble on a Kaggle GPU (~1 h)
uv run more-than-a-hack-2027 tune-seq --logprobs .cache/seq_heldout/lp_seq_v3_*.npz --out docs/TUNING_V3.md
scripts/check_reproducibility.sh        # clean copy -> predict twice, validate, retrain model, tests
uv run pytest -q
```

## Context

The task is to label all 199,319 frames of the tactical video of BAR–ATH with `H`/`A`/`D`/`X`. The board ranks on **spellF1**: predicted possessions are matched one-to-one with true ones by temporal IoU, and F1 is averaged over IoU 0.3–0.7. **macroF1** breaks ties. The floor to beat is `baseline_smooth50` at spellF1 0.2433 on the full match. It assigns each frame to the team of the nearest player and then takes a ±2 s majority vote.

**Deadline Nov 17, 2026; we submit Nov 16.** There are three deliverables: `<team>.csv`, this public repo (it must regenerate the file byte-for-byte), and a short technical document (`<team>.pdf`/`.md`).

What we have: tracking (ATD) with 46 track slots, team membership only and no player IDs. We also have 638 Smart Tagging phases, a homography (tactical, valid on every frame), and both videos.

### Facts checked against the data

These differ from the docs or add to them.

- Player speed is in **m/s**, although the XML labels it kph. The ball speed channel is almost always NaN, so we compute ball velocity ourselves.
- The ATD has no ball-owner or ball-state fields: positions and speed only.
- Barcelona (home) attacks **+x in the first half** and −x in the second. The goalkeeper positions confirm what the XML says.
- **The ball is missing in long blocks, not flickers.** There are only 149 gaps in the graded frames (median 11 s, p90 34 s), and 99.9% of missing frames sit in gaps longer than 5 s. Interpolation is useless. While the ball is missing, team evidence has to come from player shape.
- The ball track is smooth when present: only 21 frames are more than 1 m off a 5-frame median. Apparent speeds above ~40 m/s are **balls in the air** projected onto the ground plane, which is a useful "aerial ball, no control" cue.
- The ball is observed **outside the pitch** on 3,949 graded frames, in 80 episodes (58 touchline, 22 goal line). The 22 goal-line episodes equal exactly the tagged corners (12) plus goal kicks (10).
- The public GT contains **short possessions**: 0.8 s, 1.3 s, 2.8 s, 3.2 s and 3.5 s. Dead stretches last 5–20 s.

## Design rules that follow from the metric

These come from `evaluation/possession_metrics.py`.

1. **Never write `X` on a graded frame.** spellF1 forward-fills an `X` with the previous label anyway, and macroF1 counts it as a miss. `submission.write()` enforces this.
2. **Segment count dominates.** A possession is a maximal H or A run, and D splits runs. A one-frame flicker creates two fake spells. A missed D between two same-team spells merges them, and a spurious D splits one. So the output must be **decoded as segments**, not voted per frame.
3. **Boundaries need to be sharp, not smoothed.** An offset of ±1 frame is free and 3 frames still scores 0.98, but 1 s scores 0.65. Snap transitions to detected events: control touches, ball-out moments and restarts.
4. **Use soft duration priors, not hard minimums.** Sub-second possessions are real, and missing one merges its two neighbours.
5. **The GT means "last team in control".** Loose balls, passes in flight, deflections and aerial duels keep the previous team.
6. **Rules compliance.** Never tune or choose between variants on the public 5 minutes; it is a correctness check only. Use no frame-specific constants: detect kickoffs and restarts algorithmically. Do no hand labelling of this game. Pseudo-labels produced by an algorithm on this game are fine.

## Architecture (`src/more_than_a_hack_2027/`)

Every stage is `stage(ctx, cfg) -> np.ndarray` with a leading axis of 199,319 frames, indexed by video frame. The full contract is in the `data/context.py` docstring.

| module | owner | state | responsibility |
|---|---|---|---|
| `data/atd.py` | P1 | ✅ | Parse the raw ATD (~2 s) into `.cache/` `.npy` files, memory-mapped. Metres, m/s. Asserts the frame ids are 1..n. |
| `data/context.py` | P1 | ✅ | `Context` dataclass: positions, ball, team of each slot, halves, attack direction, `half_of_frame`. |
| `stages/nearest_player.py` | P1 | ✅ | Bit-exact port of `starter/baseline.py`. Also returns the distance and the slot. |
| `stages/decode.py` | P1 | ✅ v1 | Viterbi over {H, A, D} with transition penalties, run per half; `viterbi_min_dead` forces D runs ≥ 1 s. Next: duration priors for H/A, boundary snapping. |
| `submission.py` | P1 | ✅ | Writes the CSV, refuses graded `X`, re-validates with the official reader. |
| `pipeline.py`, `__init__.py` | P1 | ✅ | Methods `baseline`, `hmm-nearest` (an untuned wiring test) and `v1` (dead ball + nearest-player team evidence). CLI. |
| `data/tagging.py` | P2 | ✅ | Smart Tagging to frame intervals. Tag team names map to H/A by comparing with `ctx.team_names`. |
| `stages/ball.py` | P2 → P3 | ✅ cleaning | Ball cleaning (median spike filter), velocity, out-of-bounds distance, aerial flag. Touch detection is next (P3). |
| `stages/deadball.py`, `stages/hmm2.py` | P2 | ✅ v1 | Per-frame dead-ball evidence plus a refined list of stoppages (see below). |
| `stages/restart.py` | P2 | ✅ v1 | Restart frame and the team taking it, for each stoppage. |
| `stages/control.py` | P3 | ✅ v1 | Per-frame H/A log-odds from a team-symmetric logistic model on windowed ball/player features, trained on the external games (see "Control design"). |
| `shape.py` | P5 | — | Possession from team shape while the ball is missing. Trained on algorithmic pseudo-labels from `stages/control.py`. |
| `tag_prior.py` | P5 | — | Smart Tagging team priors. Attacking codes mean that team has the ball; DEFENDING / DEFENSIVE TRANSITION mean the other team does. |
| `external/metrica.py`, `external/degrade.py`, `validation/harness.py` | P4 | — | External validation harness (below). |
| `validation/ablation.py`, `scripts/check_reproducibility.sh` | P1, P4 | — | M3: held-out ablation table (`ablate` → `docs/ABLATION.md`) and the clean-copy reproducibility check. |
| `validation/tune.py` | P4 | ✅ | Chooses v2's `w_control` and switch penalty: 3 seeds × 3 games, held out, under two label readings (`tune` → `docs/TUNING.md`). |
| `seq/network.py` | P3 | ✅ v3 | The sequence model's inputs (the features above per frame, plus touch channels), its numpy forward pass, team-swap averaging and the ensemble average. Prediction needs numpy only. |
| `seq/training.py`, `seq/tuning.py` | P3, P4 | ✅ v3 | Training in torch (CPU or GPU), and the held-out decoder choice (`tune-seq` → `docs/TUNING_V3.md`). |
| `external/dfl.py`, `external/skillcorner.py` | P4 | ✅ | Open DFL and SkillCorner matches as extra training data for v3, with labels made to follow the Metrica rule. |
| `scripts/kaggle_seq.py` | P3 | ✅ | Builds the Kaggle kernels that train v3's networks (held out, or the shipped ensemble). |
| `diagnostics.py` | all | — | GT-free checks and debug overlays. |

Organiser modules (`games.py`, `starter/`, `evaluation/`) are imported through `_paths.ensure_repo_on_path()` and are **never modified**.

## Dead-ball design (P2)

### What the data shows

These observations come from the tracking, the tags, and the public GT, which we use only to understand the convention.

- **Convention.** D starts at the whistle, or at the frame the ball crosses the line. D ends at the restart kick or throw.
- **Collective player speed is the strongest signal.**
  - In a stoppage it is a long, smooth, slowly decaying plateau at about 0.5–1.3 m/s.
  - In play it oscillates between 1 and 5 m/s, even during slow build-up.
  - The ball is usually missing or out during stoppages. In live slow phases it keeps being passed.
- **The ball moves during stoppages.** Players knock it back to the restart spot, so ball motion alone does not mean "alive".
- **Speed alone overlaps.** Smoothed team speed has a median of 1.5 m/s shortly after a ball-out and 2.3 m/s during fast passes.
- **Smart Tagging set pieces are weak timing priors.**
  - The windows are 35 s long (5 s for kickoffs), and the restart sits anywhere from before the window starts to ~10 s into it.
  - Only 5 free kicks are tagged, so fouls are mostly untagged.
  - Throw-ins have Team = N/A.
- **Kickoff formation is detectable but not by the ball.**
  - Before kickoffs, 90–95% of observed players stand in their own half (tolerance 1 m, ~21 players observed). The ball is usually unseen until the kick.
  - The thresholds must be ~0.9, and they must not require a visible ball.

### Approach

1. **Certain-dead anchors.**
   - Ball beyond a line by ≥0.3 m, in episodes of ≥5 frames or with ≥0.5 m excursion. Dead from the crossing frame for at least 2 s.
   - Every frame of a half before its first detected kickoff. In the second half, the graded frames before kickoff are all D.
   - Goal-line outs between the posts (goals) until the next detected kickoff.
2. **A semi-supervised two-state HMM (live/dead), learned by EM on this game's own tracking.**
   - Emissions: log of team speed over ±1 s, log of the std of team speed over ±3 s, ball seen, ball moving >4 m/s.
   - The anchors are clamped to dead.
   - No labels are involved. The same model can later be fitted with supervision on the external games.
   - Output: emission LLR and the forward-backward posterior P(dead).
3. **Stoppage refinement.** Turn the posterior into intervals, then:
   - Snap starts to the ball-out crossing, or to the steepest team deceleration (whistle) minus a reaction lag.
   - Snap ends to the last restart event: a kick from a still ball with a player within 2 m, the ball re-entering from outside (throw-in), or the ball reappearing.
4. **Restart team (`stages/restart.py`), as summed log-odds of:**
   - the nearest player at the restart;
   - the Smart Tagging team, when not N/A;
   - for ball-outs, the **opponent of the last toucher** (this holds for throw-ins, corners and goal kicks);
   - after a goal, the team defending that goal;
   - at a kickoff, the team with a player inside the centre circle.
5. **Decoder input.** Strong ±D evidence inside and outside refined stoppages, scaled by confidence, plus restart-team evidence for the first ~1 s after each restart.

### What v1 implements, and what changed during the build

- **Kicks are ramps, not steps.** The ball track is already smoothed by the tracker, so a kick is a ~1.5 s rise in speed. A kick is defined as the upward crossing of 5 m/s by a ball that was below 1.5 m/s at some point in the previous 1.5 s, with a player within 2 m recently.
- **Lofted balls fake ball-outs.** A high ball near a touchline projects outside the pitch for a few frames. Out episodes are rejected if the ball is back in play within 2 s, or if they start with an aerial flag.
- **Tempered posterior.** EM fits on every 5th frame, so emissions are divided by 5 when evaluated at 25 fps. Without that, each observation counts five times.
- **Goals come from kickoffs.** The ball is rarely tracked into the net, so a mid-half kickoff is what marks a goal. All four found agree with the "Kick Off (After Goal)" tags (~3.9 s after each tag starts).
- **Restarts never land inside a certain-dead anchor** or before the posterior went dead. Refined stoppages shorter than 2 s are dropped, kickoffs excepted.
- **Found on the external games (P4), both invisible on BAR-ATH:**
  - A goal-line out between the posts (a shot just wide) late in a half was "confirmed" as a goal by the *next half's* kickoff, which anchored the rest of the half as dead (120 s of open play on Metrica game 3). A goal now needs a mid-half kickoff in the same half, with no other ball-out in between.
  - "Dead until the first kickoff" had no time limit. When the ATD is missing most players at the opening kickoff (3% of BAR-ATH graded frames have < 10 players), the kickoff is not detected and the anchor ran to the next goal's kickoff. It now applies only if that kickoff is within `pre_kickoff_max_s` (300 s) of the half start.
  - BAR-ATH output of the `deadball` command is unchanged by both fixes.

### GT-free diagnostics (`more-than-a-hack-2027 deadball`)

Current v1 values are shown next to the prior used to judge them.

| check | prior | v1 |
|---|---|---|
| stoppages | ~100–130 in a pro match | 144 (goal 4, half start 2, goal line 12, touchline 25, whistle 101) |
| dead share of graded time | ~45–50% (LaLiga ball in play ~53–56 min) | 52% |
| stoppage duration | — | p10 4.2 s, median 12.2 s, p90 48.5 s |
| set-piece tags with a restart nearby | high | 55/59 |
| validated ball-outs inside a stoppage | all | 62/62 |
| EM states, learned without labels | — | live 2.31 m/s, ball seen 92%; dead 1.01 m/s, ball seen 27% |

**Public 5-minute correctness check (M1, logged, not used for any decision):** dead-ball F1 0.81 (baseline 0.32), live/dead F1 0.95. The window holds only 17 possessions, and the scorer's bootstrap band there is ±0.32, so this confirms the wiring and nothing more.

### Open items for P2

- Tune every threshold on P4's external games with `harness.logo_search`. Top candidates: `posterior_threshold`, `restart_lag_frames`, `whistle_lag_frames`, `min_stoppage_s`, and the restart-team weights. `pipeline.run_v1` takes nested overrides: `{"deadball": {...}, "restart": {...}, "ball": {...}, "switch": ...}`.
- Restarts with `restart_kind = posterior` (42 of them) have no event to snap to. Check whether a team-acceleration onset snaps them better.
- 20 restarts have disagreeing sources (taker vs tag vs last toucher). Inspect a sample with the homography overlays.
- v1 produced 456 possessions over the match because of the placeholder nearest-player team evidence; v2 (with `stages/control.py`) produces 252 (290 before the Nov 1 re-freeze).

## Control design (P3)

### Why a learned model

On the degraded external games the nearest player is the right team on only 74–76% of live frames: passes, crosses and clearances fly past opponents, and the ground truth ("last team in control") does not switch on a duel or a deflection. Two ceilings, measured with v1's decoder:

- the true team on every frame v1 calls live → spellF1 +0.13 to +0.16 per game;
- perfect dead-ball calls with v1's team evidence → +0.01.

So team evidence was the lever. Rather than hand-tune a control state machine, `stages/control.py` learns how to read the features from the external labels, under the same leave-one-game-out protocol as every other decision.

### What v1 implements

- **Features** (18; all written from the home side; windows computed inside a half):
  - closeness: log-ratio of the two teams' nearest distances to the ball, now and as the minimum over past/future 0.5, 1 and 2 s;
  - carrying: a player within 2 m whose velocity matches the ball's (relative speed < 3 m/s), so the ball moves *with* him, as window fractions over past/future 1 and 3 s;
  - last/next carry: the team of the latest (next) carrying frame, decayed over 5 s;
  - ball velocity and position along the home attack direction; team centroid push; players within 5 m.
- **Team symmetry by construction.** Every feature flips sign when the teams are swapped (tested), and there is no intercept. The model therefore cannot learn "home has the ball more", which would not transfer between games.
- **Fit.** L2 logistic regression by Newton's method in numpy: deterministic, so `train-control` reproduces `models/control_v1.json` byte for byte. Training rows are the live, ball-visible frames of the three **degraded** Metrica games, every 3rd frame (85k rows). The weights are almost identical across the three leave-one-out folds. Retrained on Nov 1 on the corrected labels (see "External validation"). The largest weights are now near-past 2 s (+1.21), carry-future 3 s (+1.20), carry-past 3 s (+1.16), near-now (−0.92, which offsets the windowed terms) and ball velocity toward goal (+0.82). The carry windows gained the most (+0.79 / +0.63 before): once one-touch plays stop counting as possessions, carrying the ball is what control looks like. Retraining on the corrected labels moves the held-out score by less than 0.001 at fixed decoder settings (seed 0, switch 25). The labels change the best switch penalty, not the model.
- **Evidence.** The decoder gets `w_control · log P(team)` per frame, and 0 where the ball is unseen (the decoder then carries the last team, as the GT does).
- **Decoder interaction.** A grid search exposed that H→A could be bought through a one-frame D for stop + restart + 6. That capped the switch penalty at ~10, so higher values had no effect. `decode.viterbi_min_dead` forbids D runs shorter than 1 s, which real stoppages never are. It is checked against brute-force enumeration.

### Selection and results (degraded external games, leave-one-game-out)

**Current (Nov 1 evening):** the same search on games that also lose the ball in live play picks w_control 3.2 / switch 70, a tie with the setting below; held-out estimate 0.669. See "Live-play ball gaps" and [`docs/TUNING.md`](TUNING.md).

**Before the live-play gaps (Nov 1, corrected labels).**

- **Search.** `tune` runs a grid of `w_control` ∈ {0.8, 1.6, 3.2} × `switch` ∈ {12, 25, 35, 50, 70, 100}: 3 seeds × 3 games, each game scored by a model that did not see it.
- **Criterion.** The choice maximises the mean spellF1 under the two label readings: *main* (both rules) and *alt* (`loss` only, which counts one-touch plays). We cannot tell which one the organisers' GT follows. The old labels are reported but never used to choose.
- **Chosen: w_control 1.6, switch 50.** It sits at the top of a flat ridge: w3.2/s100 ties, and four more configs are within 0.004.

  | | main | alt | old |
  |---|---|---|---|
  | **w1.6 / s50 (chosen then)** | **0.720** | 0.630 | 0.590 |
  | w1.6 / s25 (Nov 1) | 0.695 | **0.639** | **0.599** |
  | best on main alone (w1.6 / s70) | 0.724 | 0.618 | 0.576 |

  The new setting gains +0.025 if one-touch plays are not possessions, and loses 0.009 if they are.
- **Held-out estimate of the procedure** (pick on two games, score the third): **0.719** main, 0.628 alt. For reference, v1 scores 0.459 and the baseline 0.373 on main labels (seed 0).
- **BAR-ATH (GT-free):** 252 possessions (Nov 1: 290), mean 12.1 s.
- **Public 5-minute correctness check (logged after the choice, not used for it):** spellF1 0.606, macroF1 0.868 (Nov 1 settings 0.588 / 0.863; bootstrap band ±0.32).

**Superseded (Oct 31, old labels).**

- Grid over `w_control` ∈ {0.4, 0.8, 1.6, 3.2} × `switch` ∈ {6, 12, 25, 50}. The score depended on switch / w_control (~15), and w1.6/s25 was chosen.
- Held-out spellF1 0.596, against v1 0.439 and baseline 0.352.
- BAR-ATH: 290 possessions. Public check 0.588 / 0.863.

### Where the error is now (held-out v2, degraded, corrected labels, 3 seeds)

| GT possession length | GT spells / game | recall @ IoU 0.5 | predicted / game | precision |
|---|---|---|---|---|
| < 1 s | 7 | 0.03 | 1 | 0.12 |
| 1–2 s | 12 | 0.29 | 11 | 0.38 |
| 2–4 s | 38 | 0.44 | 33 | 0.52 |
| 4–8 s | 51 | 0.63 | 50 | 0.65 |
| 8–16 s | 63 | 0.84 | 60 | 0.88 |
| > 16 s | 75 | 0.92 | 73 | 0.92 |

Ceilings (seed 0, v2 at 0.717). Each oracle replaces part of v2's output with the truth:

| oracle | spellF1 | gain |
|---|---|---|
| true team on every frame v2 calls live | 0.902 | +0.185 |
| GT spells < 4 s | 0.785 | +0.068 |
| GT spells < 2 s | 0.749 | +0.032 |
| perfect dead-ball calls | 0.735 | +0.018 |
| a perfect "turnover here" detector (±0.5 s, weight 5) | 0.726 | +0.009 |
| true team on live, ball-unseen frames | below v2 | — |

- **Short possessions are no longer the lever.** The "+0.15 from spells under 2 s" measured on the old labels was mostly label artefact.
- **Where the gain is now:** team calls on 2–8 s possessions (recall 0.44–0.63, mostly boundary timing and medium-length turnovers), then dead ball.
- **Ball-unseen frames cannot be tested here.** They are 0.8% of GT-live frames on Metrica but 6.7% of v2-live frames on BAR-ATH. The oracle shows nothing on the external games because there is almost nothing there to fix. Writing truth into those few frames without decoding even fragments spells, which is why it lands below v2. Before `shape.py` can be measured, `external/degrade.py` must also drop the ball during live play, at BAR-ATH's rate.

### Open items for P3

- **Boundary timing:** snap H↔A boundaries to the touch frame. Spells of 2–8 s are often found but miss the IoU threshold. This is now the clearest lever.
- **Turnover evidence:** still worth trying, but its measured ceiling is small now (+0.009 for a perfect detector added to the decoder).
- More training data: several degradation seeds as augmentation (cheap), then check the held-out gain.
- ~~Feature ablation table for the tech doc~~ done: [`docs/ABLATION.md`](ABLATION.md).

## Sequence model (v3)

### Why

v2 stalled at ~0.67 held-out spellF1. Its control model's team accuracy stays at ~0.92 whatever the model class (gradient boosting on the same features gained < 0.01), and the dead-ball stage reads a missing ball as a stoppage. What is missing is the order of events: who had the ball before, who has it next, how the players move around a gap. A network with a ±12 s view of the match can learn that from the geometry directly.

### What v3 implements

- **Inputs** (`seq/network.py`, 123 channels per frame, all GT-free). Ball state (8 channels). Per team, the 3 players nearest the ball, how many players are within 2/5/10 m, and the team's centroid and speed (26 channels each). The dead-ball stage's outputs and windowed speed and visibility statistics (33). control.py's 18 features. **Touch channels** (6 per team): the ball's owner is the player closest to it within 2 m, followed by track slot; the channels give the current spell's length, the time since the last spell and until the next, those spells' lengths, and how many different players owned the ball within ±3 s.
- **Network:** a temporal CNN of ~220k weights.
  - A strided convolution takes the 25 Hz input down to 5 Hz.
  - Six residual blocks with dilations 1–32 give a ±12.6 s receptive field.
  - A transposed convolution goes back to 25 Hz, plus a full-rate skip convolution.
  - The head outputs log P(H), log P(A), log P(D) per frame.
- **Team symmetry:** the inputs have an exact team swap (teams exchanged, pitch rotated). Training sees both copies, and prediction averages the network on the frame and on its swap, so the output is exactly team-symmetric.
- **Prediction needs numpy only.** The weights ship as JSON, and torch is used only to train.
- **Decoder:** the same Viterbi with a 1 s minimum stoppage, on 0.5 × the log-probabilities. Settings: switch 10, stop 25, restart 25, chosen held out ([`docs/TUNING_V3.md`](TUNING_V3.md)).
- **The shipped model is an ensemble:** 8 networks of the same recipe, 7 training seeds plus one variant that learns the extra data through its own output head. Their log-probabilities are averaged (the renormalised geometric mean of the probabilities).

### Training data

Every match is degraded like the Metrica games (`external/degrade.py`, ball and player losses at BAR-ATH's measured rates) and labelled to follow the same rule: possession changes when the other team *gains* control.

- **Metrica sample data**, games 1–3 (github.com/metrica-sports/sample-data), 6 degradation seeds. The only data used to choose anything.
- **DFL open data (IDSSE):** 7 Bundesliga matches with 25 Hz positions, 6 degradation seeds.
  - Citation: Bassek, Rein, Weber et al., "An integrated dataset of spatiotemporal and event data in elite soccer", *Scientific Data* 12, 195 (2025), doi:10.1038/s41597-025-04505-y, CC BY 4.0.
  - Labels: DFL's per-frame possession flag changes on every touch. A change counts only if the new team has a control event of its own within ±1.5 s (`external/dfl.py`).
- **SkillCorner open data:** 20 A-League 2024/25 matches of broadcast tracking (github.com/SkillCorner/opendata, MIT licence, credit SkillCorner), resampled 10 → 25 Hz, 3 degradation seeds.
  - Labels: possessions are runs of SkillCorner's phases of play, with the Metrica one-touch rule applied (`external/skillcorner.py`).

The extra matches fill half of every training batch and train the same output head as Metrica. Every score below is on a Metrica game whose networks never saw it.

### Training on a Kaggle GPU

`scripts/kaggle_seq.py` builds a Kaggle kernel from this checkout's `seq/network.py` and `seq/training.py`. The kernel reads private Kaggle datasets that hold the cached inputs above. Those are built from the open data only: never from `data/`, which may not be redistributed, and with no organiser code. The degraded copies replay BAR-ATH's per-frame missing-player counts, a list of integers with no tracking.

- **Recipe:** 3200 steps of 12 crops × 160 s; AdamW, one-cycle learning rate 2e-3, weight decay 0.05, dropout 0.3, input noise 0.1; extra loss weight on possessions under 4 s (×2) and within ±12 frames of a possession change (×1).
- **Speed:** ~5 min per network on a T4.
- **`v3-heldout`** trains one network per held-out Metrica game and writes held-out log-probabilities, which `tune-seq --logprobs` and `ablate` score locally.
- **`v3-final`** trains on all three games and writes `models/seq_v3_*.json`.

### Results (held out, degraded with live-play gaps, Metrica games 1–3, 3 evaluation seeds)

How v3 got there. Each row is the leave-one-game-out estimate under the main labels, with the decoder chosen held out for that row. Rows before the last are single networks.

| step | spellF1 |
|---|---|
| v2 (control model + dead-ball stage) | 0.669 |
| v3, Metrica games only | 0.816 |
| + DFL matches, learned through a second head | 0.831 |
| + DFL through the main head, 6 degradation seeds, 3200 steps | 0.835 (training seeds 0.826–0.837) |
| + SkillCorner matches (one-touch rule) | 0.837–0.839 |
| + touch channels, one network | 0.830–0.839 (own-head variant 0.824) |
| **ensemble of the 8 touch networks (shipped)** | **0.845** (alt labels 0.704) |

- **Training seeds matter as much as most changes:** one network varies by ±0.005 with its seed. The touch channels alone are within that noise. Averaging networks is what gains: the ensemble beats its mean member by +0.010 and its best by +0.006.
- **Picking the best of many ensemble subsets would be selection bias.** Subsets of the networks trained along the way scored 0.829–0.846. The shipped ensemble is every network of the final recipe, not a chosen subset.
- **Tried, no gain** (each within ±0.005 of its reference):
  - restart-team evidence on top of v3;
  - rescuing short spells after decoding;
  - 6400 steps;
  - 70% extra data per batch;
  - 96 channels;
  - a 7th dilation (64);
  - a stride of 2 instead of 5;
  - ending training with a Metrica-only phase.

**At the shipped decoder settings** ([`docs/ABLATION.md`](ABLATION.md), paired bootstrap against v2): the ensemble scores 0.845 (seed sd 0.002), +0.176 [+0.155, +0.198]. One network (training seed 0) scores 0.839, +0.170 [+0.150, +0.190]. macroF1 rises 0.878 → 0.959 and team accuracy 0.923 → 0.963, with 233 possessions per game (true 246).

- **Under the alternative label reading** (one-touch recoveries count), v3 scores 0.704. v2's 0.595 under that reading is the comparison.
- **Clean tracking:** v2 gains +0.107 from clean tracking, so input quality is still the largest remaining cost. v3 has not been measured on clean tracking.

## External validation: the only place we tune

- **Data.** Metrica's public sample data (github.com/metrica-sports/sample-data); their README asks that public use acknowledges the source, so the tech doc must. It is the only data any choice is made on. v3 also *trains* on open DFL and SkillCorner matches (see "Sequence model (v3)", which gives the citations the tech doc must carry).
  - Games 1 and 2: CSV tracking (one file per team) plus `RawEventsData.csv`.
  - Game 3: EPTS tracking plus JSON events. The player order changes at every substitution (11 format specs), so `external/metrica.py` gives every player one fixed slot.
  - All 25 fps, 105 × 68 m. `external/metrica.py` downloads them once into `data/other_matches/metrica/` and keeps them there (never deleted; `data/` is never committed), and caches the parsed arrays in `.cache/external/` (~3 s for all three).
- **Labels** (`metrica.labels_from_events`). Events are converted to frame-level H/A/D. The tech doc must say these come from Metrica analysts' event data, cover 3 games, and are not the provided game.
  - Stops: `BALL OUT` and shots that go out or in (end frame), fouls (`FAULT RECEIVED`, `CHALLENGE …-FAULT-…` without advantage), offside and hand ball, end of half.
  - The ball stays dead until the next **tagged** restart: a `SET PIECE` not marked RETAKEN, or a goal kick. Between a whistle and the restart the stream often has 1–3 "after the whistle" actions within ~3 s at walking pace (a ball kicked away, a shot after the flag), so those do not restart play. A few restarts are never tagged; that shows as ≥ 3 play events spanning ≥ 5 s before the next tag, and play then resumes at the first of them. `RECOVERY`, `CHALLENGE` and `CARD` never restart play.
  - In play, possession changes when the other team **gains** control: its PASS, CARRY, SET PIECE, SHOT or RECOVERY. Challenges never switch it. Since Nov 1, two label rules (`metrica.LABEL_RULES`) follow CHALLENGE.md's list of what does not count:
    - **`loss`: a BALL LOST never starts a possession.** Metrica tags both sides of a tackle or interception. The loser's BALL LOST often lands 1–35 frames *after* the winner's RECOVERY, and counting it handed the ball back to the team that had just lost it. This follows from the definition alone.
    - **`one_touch`: a one-touch RECOVERY is not a possession.** That is a recovery whose team's next own action is a loss (BALL LOST or BALL OUT) within 2 frames: a header or clearance straight back to the opponent or out of play. 232 of 233 such losses are tagged on the same frame as the recovery, and the next shortest gap is 3 frames. The clean tracking agrees: after such a recovery the ball stays within 1.5 m of that team for a median of 6 of the next 25 frames, against 16–25 when the team keeps it. A CARRY of ≤ 2 frames is not a carry either (game 3's export adds a 1-frame CARRY to each one-touch recovery).
    - **Why it matters:** these artefacts were most of the "short possessions" in the old labels. Before the fix, an oracle on GT spells under 2 s was worth +0.15 spellF1, and the control model's sign was right on only 34% of frames inside GT spells under 1 s: they were not there to be found.
    - **What is uncertain:** where the organisers' tracking-derived GT draws the one-touch line cannot be checked from here. Decoder settings are therefore chosen to hold up under both `("loss", "one_touch")` and `("loss",)` alone (`validation/tune.py`, [`docs/TUNING.md`](TUNING.md)).
  - Each period is graded from its first tracking frame, dead until the kickoff (like BAR-ATH's second half). The last frame of period 1 is X so the metric sees two runs.
  - Result (both rules): dead share 32–45%, 225–258 spells per game (median 8.6–10.3 s; 5–9 under 1 s), 97–100 stoppages. The older labels had 350–401 spells (median 4–5 s); `rules=()` still builds them.
- **Degradation** (`external/degrade.py`), calibrated on the competition **input** only:

  | statistic | BAR-ATH ATD | Metrica clean | degraded |
  |---|---|---|---|
  | ball missing (graded) | 40.7%, 99.7% of it in gaps > 5 s | 32–41%, ~97% in gaps > 5 s | unchanged (already alike; in both the ball is mostly missing while dead) |
  | ball accel. autocorrelation lag 1 / lag 5 | 0.989 / 0.859 | −0.45 / 0.00 (piecewise linear, instant kicks) | 0.99 / 0.82–0.84 |
  | ball \|accel\| p50 / p90 / p99 (m/s²) | 5.7 / 18 / 37 | 0.8 / 6.5 / 286 | 3.3–4.2 / 17.5 / 29–33 |
  | players per frame | 20.5 (p5 14), not spatial | 22.0 | 20.4–20.6 |
  | track slots | fragments, no IDs | one per player | a fresh slot per visible run |

  - Ball: 0.5 m white noise, then a Gaussian smoother (σ = 7 frames), both picked by a grid fit to the ATD statistics above. Each visible run is padded by linear extrapolation first, so a ball that crosses a line and disappears is not smoothed back inside (that had cut detected outs from 77 to 16).
  - Players: the ATD's own per-frame missing-player count is replayed from a seeded offset; who is missing follows a per-player priority redrawn every ~35 s (the ATD's median track run). The series is read from the ATD at run time, never stored in the repo.
  - Known residuals: smoothed peak ball speed is lower (p99 19 vs 25 m/s), and Metrica's ball is almost never seen while dead (0.5–15%, the ATD's EM estimate is 27%).
- **Protocol** (`validation/harness.py`). `evaluate()` scores any method with the official `possession_metrics.evaluate`. `logo_search()` does leave-one-game-out selection: for each held-out game it picks the config with the best mean spellF1 (macroF1 tie-break) on the other two, and reports the held-out score. Every hyperparameter choice is made here. Use the harness for relative decisions: event-derived labels are not Metrica's corrected-tracking ownership.
- **Public 5 min.** Run `evaluation/run_evaluation.py` at milestones as a correctness alarm only, and log the result. Never pick a variant on it.

### First results (Oct 31, seed 0; untuned v1; old labels, superseded)

| method | variant | spellF1 (g1 / g2 / g3) | mean spellF1 | macroF1 | live/dead F1 | team acc. | spells (GT 375) |
|---|---|---|---|---|---|---|---|
| baseline smooth50 | clean | 0.430 / 0.412 / 0.389 | 0.410 | 0.827 | 0.925 | 0.875 | 502 |
| v1 | clean | 0.506 / 0.544 / 0.505 | **0.518** | 0.874 | 0.957 | 0.879 | 377 |
| baseline smooth50 | degraded | 0.383 / 0.351 / 0.322 | 0.352 | 0.796 | 0.925 | 0.825 | 533 |
| v1 | degraded | 0.451 / 0.447 / 0.419 | **0.439** | 0.846 | 0.958 | 0.833 | 405 |

- **The degraded variant transfers.** It lands close to the public BAR-ATH check (v1 0.441, baseline 0.326, live/dead 0.948). That check was not used to build or tune anything; the agreement is evidence that the degradation is realistic, not a target.
- Seed sensitivity: seed 1 gives a v1 degraded mean of 0.439 too (±0.01 per game). Decide on means over games, and add seeds before trusting gaps below ~0.02.
- Where the remaining error is: live/dead is at 0.96 on both variants, while team accuracy drops from 0.88 to 0.83 when degraded. Possession *team* evidence (P3 `stages/control.py`, P5 `shape.py`) is the lever now, not dead ball.

### Live-play ball gaps (Nov 1)

- **What changed.** Metrica's ball is visible on ~99% of live frames; BAR-ATH's on ~93% of the frames v2 calls live. `degrade.live_gaps` now replays BAR-ATH's three kinds of live-play gap (`degrade.LIVE_GAPS`): 10 losses of 5–11 s inside live play per 50 live minutes, a loss just before 27% of stoppages, and a late pick-up after 6% of restarts. They are measured against v2's own live/dead calls on BAR-ATH (GT-free); placing them on a Metrica game uses its labels, which only says where play is live. Gaps only hide the ball.
- **Cost:** −0.044 spellF1 at fixed settings (0.720 → 0.676), all of it through the dead-ball stage. It reads a missing ball as a dead ball and cuts possessions in two: on seed 0 it labels ~9,000 live frames per game D (against ~2,000 the other way), through ~19 false stoppages per game (median 7 s) and stoppage edges that start up to 10 s early or restart up to 5 s late.
- **Ceilings with gaps** (seed 0, held out): perfect team calls on the frames v2 calls live 0.843; perfect dead-ball calls 0.718; a perfect team model on ball-unseen live frames +0.009 at most, so the planned `shape.py` waits.
- **Re-tuned** on the gapped games (`tune`, [`docs/TUNING.md`](TUNING.md)): w_control 3.2 / switch 70, a tie with w 1.6 / switch 50 (+0.0002, paired s.e. 0.003). The rule fixed before the run picks the top of the table, so it ships. Leave-one-game-out estimate 0.669 main, 0.595 alt. The control model was retrained on the gapped games (the reproducibility check requires it).
- **BAR-ATH:** 260 possessions (252 before). Public 5-minute check, logged after the choice: spellF1 0.588, macroF1 0.862 (0.606 / 0.868 before; band ±0.32).

### Open items for P4

- Hyperparameter search for P2's thresholds and the stop/restart penalties. Use the `validation/tune.py` pattern: 3 seeds, held out, under both label readings. The degraded variant is the decision variant; clean is a sanity check.
- ~~Multi-seed selection and paired bootstrap~~ done: `validation/tune.py` selects over 3 seeds, and `validation/ablation.py` gives paired intervals (row "decoder as frozen on Nov 1").
- ~~Ball dropout in live play~~ done (Nov 1): see "Live-play ball gaps".
- Fit duration priors (spell and stoppage lengths) on the external labels for the HSMM (P1).
- Optionally supervise the dead-ball HMM on the external labels, and check that it beats the unsupervised EM under LOGO.

## Freeze and reproducibility (M3)

### What is frozen

- **Method:** `pipeline.SUBMISSION_METHOD = "v3"`, with `V3_DEFAULTS` and the shipped ensemble `models/seq_v3_*.json` (8 networks, trained by `scripts/kaggle_seq.py v3-final`). The inputs also use `models/control_v1.json` for control.py's features. `uv run more-than-a-hack-2027 predict --out evaluation/results/<team>.csv` runs exactly this.
- **Output today (re-frozen Nov 2 on v3):** 218 possessions (mean 17.6 s), sha256 `f26b32ba809b8d7d258f0426c234a4d01f2c604b54fafada1f5657dda94469d4`. `predict` takes ~85 s: 8 networks × 2 team views, numpy on CPU.
- **Public 5-minute check** (logged after the choice; correctness only, band ±0.41 with 17 possessions):

  | method | spellF1 | macroF1 | possessions (true 17) |
  |---|---|---|---|
  | v3 | 0.571 | 0.741 | 11 |
  | v2 | 0.588 | 0.862 | 15 |

  The spellF1 gap is noise. The macroF1 gap is all dead ball: v3 labels only 496 of the slice's 1,469 dead frames D (v2 1,318), while its team calls on live frames are at least as good. See the next item.
- **Known transfer risk: a visible ball during a stoppage.** v3 calls D on only 5% of the full match's graded frames where the ball is seen, against 25% for v2. Of the frames v2 calls dead and v3 calls live, 87% have the ball in view. This is GT-free and needs no labels.
  - The cause is a known residual of the harness: the ball is almost never visible while dead in Metrica's data (0.5–15%), against ~27% in the ATD (see "Degradation"). The SkillCorner inputs hide it the same way.
  - So v3 has learned "ball seen → live". The external harness cannot see the failure, which biases the v3-vs-v2 margin above in v3's favour on exactly this point.
  - Full-match dead shares: v3 39.7%, v2 52.1%. A normal match is ~40–45% dead, so v2 likely over-calls D. v3 likely under-calls it while the ball is seen.
  - Next step, before the final submission: the DFL raw data tracks the ball while dead, but `external/dfl.py` hides it on purpose (it keeps the first 0.5 s and last 1 s of each stoppage). Keep it visible at the ATD's measured rate instead, then retrain v3. Held-out DFL matches can then measure the effect, which Metrica cannot: it has no dead-ball ball.
- **Why the switch clears the rule below:** held out, v3 is +0.176 over v2 ([+0.155, +0.198], paired bootstrap in [`docs/ABLATION.md`](ABLATION.md)). That is an order of magnitude beyond the noise floor.
- **History:**
  - first freeze (Nov 1 morning): v2, switch 25 and the model trained on the old labels, 290 possessions, sha256 `ead4e48720eca1a96edb20ec5f14eadb2acdab09a25e4f3d5d506ebb93bd1212`. It cleared the rule below on the corrected labels: −0.024 [−0.031, −0.018] for the old settings;
  - after the label fix: v2, w_control 1.6, switch 50, 252 possessions, sha256 `46fa8c61b3e9d7895d42726bddeb1a2d487e369515cdabe43bc0b18a74787d17`;
  - after adding live-play ball gaps (Nov 1 evening): v2, w_control 3.2, switch 70, control model retrained on the gapped games, 260 possessions, sha256 `f1d4a1919a3e05dd354bfacc9d63b7583eb4f262c26c0eca3fc97052898a0550`. The step from w 1.6 / switch 50 was a tuning tie that followed the harness change, not an improvement (see "Live-play ball gaps").
- **Changing it** (an M2 improvement): the new method must beat v3 on the held-out external games by more than the ablation's bootstrap interval. Then set `SUBMISSION_METHOD`, rerun `ablate` and `scripts/check_reproducibility.sh`, and record the new hash here. For v3 the bar is high: ensembles of the same networks already differ by ±0.008, so a change needs several training seeds behind it (≥ 4) before it counts.

### Reproducibility check (`scripts/check_reproducibility.sh`)

It builds a clean copy of the repository: the files git would publish, or `--head` for a real `git clone` after committing. The data bundle is linked in from `data/`. In the copy it then:

1. checks for leaks: no `data/`, `.cache/`, private GT, submissions, `.npz`/`.mp4`/`.pdf`, or files over 5 MB;
2. runs `uv sync --frozen`, then `predict` twice (cold cache, warm cache) and requires identical bytes;
3. runs the organisers' `validate_submission.py`;
4. requires the same bytes as this checkout's own output (and `EXPECTED_SHA256` if set);
5. downloads the Metrica data and retrains the control model, which must equal `models/control_v1.json` byte for byte;
6. runs the whole test suite.

First run (Nov 1): 72 files, all six checks pass, and all 68 tests run in the copy (none skipped). Running it caught a regression from adding the feature-drop option for the ablation: boolean column indexing in `control.train` gave a non-contiguous array. That changed the BLAS summation order and the last bits of the weights, so the shipped model no longer regenerated byte for byte. Fixed with `np.ascontiguousarray`; the BAR-ATH output was unaffected.

Re-freeze (Nov 1, after the label fix): 75 files, all six checks pass with `EXPECTED_SHA256=46fa8c61…`, and the control model retrained in the copy equals the shipped one byte for byte. All tests run in the copy (80 at the time).

**v3's networks are not retrained by the check.** They are trained on a Kaggle GPU (`scripts/kaggle_seq.py v3-final`), and GPU training is not bit-equal to CPU training. torch's deterministic settings are on, but a CUDA op without a deterministic version only warns, so even a rerun on the same GPU can differ in the last bits. The shipped JSON weights are therefore the record. The check proves that the submission follows from them byte for byte (steps 2–4), and that the control model they read their features from still retrains exactly (step 5). Prediction is numpy only, with a fixed averaging order.

Re-freeze on v3 (Nov 2): 100 files, the 8 network files 4.6 MB each, all under the 5 MB limit. All six checks pass with `EXPECTED_SHA256=f26b32ba…`. 112 tests pass in the copy; the 6 torch training tests skip there, because the copy installs no torch: prediction needs numpy only.

### What the ablation says ([`docs/ABLATION.md`](ABLATION.md))

Held-out, degraded with live-play ball gaps, 3 games × 3 seeds, corrected labels, decoder settings fixed at v2's (w_control 3.2, switch 70):

- **Stages:** the dead-ball stage is worth +0.080 spellF1 (baseline 0.354 → 0.434 with nearest-player team evidence). The learned control model is worth +0.235 (→ v2 0.669).
- **The Nov 1 decoder (w 1.6, switch 25):** −0.013 [−0.017, −0.009].
- **Control feature groups** (proximity's interval touches 0, the rest exclude it):

  | without | Δ spellF1 |
  |---|---|
  | temporal context | −0.118 |
  | look-ahead | −0.052 |
  | carry | −0.030 |
  | ball direction | −0.018 |
  | proximity | −0.008 |
  | team shape | −0.008 |

  On the old labels, carry and team shape were indistinguishable from zero. They lost to one-touch plays that were labelled as possessions.
- **The 1 s minimum stoppage is essential:** −0.061 [−0.069, −0.053]. At switch 50, a turnover through a short fake D is much cheaper than a switch, and `viterbi_min_dead` is what closes that route.
- **Indistinguishable from zero:** restart-team evidence, +0.001 [−0.001, +0.003]. The external games have no Smart Tagging, so the part of it that comes from the tags can only matter on BAR-ATH, and the harness cannot measure that.
- **Noise floor:** v2's spread across degradation seeds is 0.007. A config difference below ~0.01 means nothing without a paired interval.
- **Clean tracking is worth +0.107** (+0.057 before the live-play gaps). That is roughly the most this pipeline can gain from better input. It matters for the Nov 8 go/no-go on video ball detection, which can recover only part of that gap.

Nothing was removed. Every component either helps or is within noise.

**v3** (rows added Nov 2): the 8-network ensemble is +0.176 [+0.155, +0.198] over v2, and one network alone +0.170 [+0.150, +0.190]. The v2 rows above still describe the features v3 reads through control.py and the dead-ball stage. v3 has no per-input ablation yet: each one needs several GPU-trained networks to rise above the ±0.005 seed noise.

## Workstreams (5 people)

| who | owns | next deliverable |
|---|---|---|
| **P1 Platform & decoder** | platform, `stages/decode.py`, final assembly, reproducibility | H/A duration priors (HSMM); event-snapping hooks |
| **P2 Dead ball** | `data/tagging`, `stages/ball` (cleaning), `stages/deadball`, `stages/hmm2`, `stages/restart` | Anchors + EM HMM + refined stoppages + restart team |
| **P3 Ball & control** | `stages/ball` (touches, aerial), `stages/control` | H↔A boundary timing: touch events, snapping boundaries (2–8 s possessions) |
| **P4 External validation** | `external/metrica`, `external/degrade`, `validation/harness`, `validation/tune`; the hyperparameter search | P2 thresholds + stop/restart penalties via the `validation/tune.py` pattern; live-play ball dropout in `external/degrade.py` |
| **P5 Ball-free + stretch** | `shape`, `tag_prior`; stretch | Pseudo-label set + shape classifier (critical: the ball is missing in long blocks) |

Everyone writes the tech-doc section for their own module. P1 compiles it.

## Milestones

- **M0, Nov 1–2.** ✅ Skeleton, interfaces and baseline parity; `.cache/` gitignored. Still to do: rename the tactical mp4 to `data/tactical/BAR-ATH_tactical.mp4`, and make sure `origin` (`meroo893/more-than-a-hack-2027`) is public before submitting.
- **M1, Nov 3–6.** v1 of every module; first fused submission; external harness running.
- **M2, Nov 7–11.** HSMM durations, boundary snapping, restart logic, the shape model, fusion weights tuned externally. **Nov 8 go/no-go for the stretch work.**
- **M3, Nov 12–14.** Feature freeze; ablation table for the doc (external harness); reproducibility check from a clean clone. ✅ The tooling is done and passes on v2 (Nov 1). At the real freeze: commit, then rerun `ablate` and `scripts/check_reproducibility.sh --head`, and record the hash.
- **M4, Nov 15–16.** Final `<team>.csv` + tech doc; validate; **submit Nov 16**.

## Stretch (P5 after M1, gated at Nov 8)

1. **Video ball detection on the tactical mp4, local GPU only.** The T&Cs forbid uploading the match video to cloud LLMs, and cloud compute is a grey area. Use a public detector trained on other games, and run it only on graded frames with no ball. Back-project to the pitch with `load_homography.unproject`, and keep a detection only if it is consistent with the trajectory. Drop the idea if it recovers less than ~20% of the gaps or shows no gain.
2. **Broadcast replay segments as a stoppage signal.** Only if time allows. It needs the broadcast aligned to tactical by matching the two ATDs.

## Verification

1. `uv run pytest -q`. It covers loader alignment, the published coverage numbers, attack direction, a bit-exact baseline port, the writer refusing graded `X`, and the decoder on synthetic data.
2. **Decision metric:** `uv run more-than-a-hack-2027 eval-external` (P4), against the baseline on degraded games; settings chosen with `harness.logo_search`.
3. **Submission:** `predict --out evaluation/results/<team>.csv`, then `python evaluation/validate_submission.py evaluation/results/<team>.csv`. Expect OK, 199,319 rows, a plausible possession count, and no graded `X`.
4. **Public check (correctness only):** `uv run python evaluation/run_evaluation.py --bootstrap 1000`.
5. **Reproducibility:** `scripts/check_reproducibility.sh` (`--head` after committing). It runs a fresh copy plus the data bundle, then `uv sync && predict` twice, validate, the model retrain and the tests, and requires identical sha256.
6. **No leaks:** the same script fails if the published files include `data/`, `.cache/`, private GT, submission CSVs or derived tracking exports.
7. **Ablation:** `uv run more-than-a-hack-2027 ablate --out docs/ABLATION.md` regenerates the table byte for byte.
8. **Tuning:** `uv run more-than-a-hack-2027 tune --out docs/TUNING.md` regenerates the decoder choice byte for byte; `pipeline.V2_DEFAULTS` must match its "Chosen" line. For v3, `tune-seq --logprobs .cache/seq_heldout/lp_seq_v3_*.npz --out docs/TUNING_V3.md` does the same from the held-out log-probabilities, and `pipeline.V3_DEFAULTS` must match its "Chosen" line.
