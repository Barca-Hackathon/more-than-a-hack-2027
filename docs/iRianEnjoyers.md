# iRianEnjoyers — More than a Hack 2027, Phase 1: Ball Possession

**Technical document.** Code: <https://github.com/Barca-Hackathon/more-than-a-hack-2027>.
The full working log, with every experiment and number, is [`docs/PLAN.md`](PLAN.md).

## 1. Summary

We label every frame of BAR–ATH with a **temporal convolutional network ensemble** that reads
per-frame features of the tracking, followed by a **Viterbi decoder** over `H` / `A` / `D` that
turns per-frame probabilities into whole possessions. The networks are trained only on **open
tracking data from other matches** (Metrica sample data, the DFL/IDSSE Bundesliga matches and
SkillCorner A-League matches). Each of those matches is first **degraded to look like the
competition's automatic tracking**. No label from BAR–ATH, public or private, is used to train,
tune or choose anything.

| | |
|---|---|
| Submission method | `v3`: 8-network ensemble + Viterbi decoder (`pipeline.SUBMISSION_METHOD`) |
| Held-out estimate (Metrica games 1–3, degraded, leave-one-game-out) | spellF1 **0.853** (alternative label reading 0.721), macroF1 0.963 |
| Previous method (`v2`, hand-designed stages + logistic model) | spellF1 0.669 held out |
| Baseline (`baseline_smooth50`) on the same harness | spellF1 ~0.35 |
| BAR–ATH output | 213 possessions, mean 16.0 s; sha256 `39e5382a8dd01638bab6ea9b34cc87c5240fa475d34754c3c0ec5970e6ea955a` |
| Public 5-minute check (correctness only, not used for any decision) | spellF1 0.593, macroF1 0.913 |

## 2. Approach, and why

### 2.1 What the metric rewards

spellF1 matches whole possessions by temporal IoU. Three rules follow from
`evaluation/possession_metrics.py` and shaped everything else:

1. **Segment count dominates.** A one-frame flicker creates two fake possessions, and a missed
   `D` merges two. So the output is **decoded as segments** (Viterbi with transition penalties),
   never voted frame by frame.
2. **Never write `X` on a graded frame.** It is forward-filled under spellF1 and a miss under
   macroF1. The writer (`submission.py`) refuses to.
3. **Possession means "last team in control".** Loose balls, passes in flight and aerial duels
   keep the previous team.

### 2.2 What the data looks like

Measured on the BAR–ATH tracking itself (GT-free):

- **The ball is missing on 41% of graded frames, in long blocks, not flickers.** There are 149 gaps
  (median 11 s), and 99.9% of the missing frames sit in gaps over 5 s. Interpolation cannot help;
  team evidence must come from how the players move.
- **No player IDs**: 46 track slots with team membership only.
- Player speed in the ATD is in m/s, although the XML labels it kph. The ball speed channel is
  almost always empty, so we compute ball velocity ourselves.
- During stoppages, team speed is a smooth plateau of about 0.5–1.3 m/s; in play it oscillates
  between 1 and 5 m/s. The ball is still seen on about 28% of dead frames, because players knock it
  back to the restart spot.

### 2.3 How we got to the final method

We kept every decision on **external games with labels**, under leave-one-game-out validation, and
moved to a new method only when it beat the old one by more than a paired-bootstrap interval.

| method | what it is | held-out spellF1 |
|---|---|---|
| baseline | nearest player to the ball, ±2 s majority vote | ~0.35 |
| v1 | unsupervised dead-ball HMM (fitted by EM on BAR–ATH's own tracking) + nearest-player team evidence + Viterbi | 0.434 |
| v2 | v1 + a team-symmetric logistic "control" model trained on external games | 0.669 |
| **v3 (submitted)** | a temporal CNN ensemble reading all of the above as inputs, + Viterbi | **0.853** |

v2 stalled because a per-frame model cannot see the order of events: who had the ball before, who
gets it next, how players move through a ball gap. A network with a ±12 s view learns that from the
geometry directly.

### 2.4 The submitted pipeline (v3)

1. **Inputs** (`seq/network.py`, 123 channels per frame, all computed from the BAR–ATH tracking
   with no labels):
   - ball state: cleaned position, velocity, seen/unseen, out-of-bounds distance, aerial flag;
   - per team: the 3 players nearest the ball, player counts within 2/5/10 m, centroid and speed;
   - the dead-ball stage's outputs (`stages/deadball.py`, `stages/hmm2.py`): certain-dead anchors
     (ball over a line, before a detected kickoff), the live/dead HMM posterior, and windowed speed
     and visibility statistics;
   - the 18 features of the v2 control model (`stages/control.py`): closeness, carrying, last/next
     carrier, ball direction;
   - touch channels: the ball's owner is the closest player within 2 m, and the channels describe
     how long the current touch spell lasts and the time to the previous and next ones.
2. **Network**: a temporal CNN of about 220k weights. A strided convolution downsamples 25 Hz to
   5 Hz, six residual dilated blocks give a ±12.6 s receptive field, and a transposed convolution
   plus a full-rate skip go back to 25 Hz. The output is log P(H), log P(A), log P(D) per frame.
3. **Exact team symmetry**: each network is run on the frame and on its team-swapped copy (teams
   exchanged, pitch rotated), and the two are averaged. The model cannot learn "home has the ball
   more".
4. **Ensemble**: 8 networks of one recipe (7 training seeds plus one variant that learns the extra
   data through its own head), with their log-probabilities averaged.
5. **Decoder** (`stages/decode.py`): Viterbi over {H, A, D} per half, with penalties switch 10,
   stop 25 and restart 25, and a 1 s minimum stoppage. The minimum stoppage closes a shortcut where
   the decoder would buy an `H`→`A` change through a one-frame fake `D`. Removing it costs 0.061.
6. **Kickoff-wait rule** (`network.kickoff_wait`): if a half starts inside a dead-ball anchor that
   lasts at least 10 s and the players never reach typical live-play speed, the wait is `D`. See
   §4.

Prediction is numpy only (weights ship as JSON) and takes about 80 s on a laptop CPU.

## 3. Assets: used and dropped

| asset | used? | how |
|---|---|---|
| Tactical ATD (tracking + ball) | **yes, the only input** | every feature above |
| Tactical ATD metadata (XML) | yes | frame alignment, halves, team names, pitch size |
| Smart Tagging | **dropped from the submission** | Used by v1/v2's restart-team evidence. Held out it was indistinguishable from zero (+0.001 [−0.001, +0.003]), and the external games have no Smart Tagging, so its effect could not be validated. v3 ignores it. |
| Homography | no | The ATD is already in pitch coordinates. It would be needed only for the video work below. |
| Tactical video | no | We measured a perfect video ball detector's ceiling held out: at most **+0.010** (filling every live-play ball gap). The T&Cs keep the video off cloud services, and a detector on a 4 GB laptop GPU would land well below that ceiling. |
| Broadcast video + ATD | no | We aligned the broadcast tracking to the tactical timeline (one offset per half, ±1 frame) and measured it. It fills 12.8% of the tactical ball gaps, but 91% of those gaps are stoppages already called dead. Held-out ceiling of fusing it: **+0.004**. Its ball is also more than 5 m from the tactical ball on 13% of shared frames. |
| Public 5-minute GT | correctness check only | scored once after each freeze, never used to pick anything |

## 4. Training and validation data

**None of it comes from the provided game.** Every label below was produced by the data providers'
own analysts or tracking systems. We did no hand labelling at all.

| dataset | source and licence | labels | amount |
|---|---|---|---|
| **Metrica Sports sample data** | <https://github.com/metrica-sports/sample-data>, courtesy of Metrica Sports | Possession labels **we derived algorithmically** from Metrica analysts' event data (`external/metrica.py`) | 3 matches. This is **the only data any setting was chosen on** |
| **DFL / IDSSE open data** | Bassek, Rein, Weber et al., *An integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data 12, 195 (2025), doi:10.1038/s41597-025-04505-y, **CC BY 4.0** | DFL's per-frame possession flag, kept only where the new team has a control event of its own within ±1.5 s (`external/dfl.py`) | 7 Bundesliga matches (J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9) |
| **SkillCorner open data** | <https://github.com/SkillCorner/opendata>, **MIT licence**, credit SkillCorner | Runs of SkillCorner's phases of play, with the one-touch rule below | 20 A-League 2024/25 matches, resampled from 10 to 25 Hz |

**Turning events into labels (Metrica).** The ball is dead from a ball-out, a goal, a foul,
offside or a hand ball until the next tagged restart. In play, possession changes when the other
team **gains** control: its pass, carry, set piece, shot or recovery. A challenge never switches it.
Two rules follow CHALLENGE.md's definition of what does not count:

- a BALL LOST never starts a possession (Metrica tags both sides of a tackle);
- a one-touch recovery, such as a header or clearance straight back to the opponent, is not a possession.

We cannot tell where the organisers' GT draws the one-touch line, so the decoder is chosen to hold
up under both readings (with and without the second rule).

**Degradation** (`external/degrade.py`). Clean external tracking is made to look like the ATD,
calibrated on the competition **input** only, never on its labels:

- the ball gets 0.5 m noise and a Gaussian smoother (σ = 7 frames), fitted to the ATD's
  acceleration statistics;
- players drop out at the ATD's per-frame missing-player rate, and each visible run gets a fresh
  track slot (no IDs);
- the ball is hidden in live play at BAR–ATH's measured rate of live-play gaps;
- during stoppages, the DFL and SkillCorner matches show the ball with the seen/unseen pattern of a
  BAR–ATH stoppage of similar length. Only the mask is copied, never a position.

**Training.** On a free Kaggle T4 GPU (`scripts/kaggle_seq.py`), about 5 minutes per network:
3,200 steps of 12 crops × 160 s, AdamW, one-cycle learning rate 2e-3. Extra loss weight goes on
possessions under 4 s and on frames near a possession change. A second, training-only head learns
which team's player is within 1.5 m of the ball in the *clean* tracking, which teaches the network
to read through the degradation (+0.007 held out). The Kaggle datasets hold cached features built
only from the open data above; the code that builds them is in this repository.

**Validation.** Every choice was made by leave-one-game-out on the three degraded Metrica games:
train on two, score the third, over three degradation seeds. Held-out DFL matches were a second
test set for what Metrica cannot show, such as a visible ball during stoppages. A change shipped only
if its paired-bootstrap interval cleared the noise floor (±0.005 per network's training seed,
±0.008 between ensembles), with the rule written down before the runs. The ablation table is
[`docs/ABLATION.md`](ABLATION.md). The decoder search is [`docs/TUNING_V3.md`](TUNING_V3.md).

## 5. The possession rules we implemented

- **Possession is "last team in control."** It changes only when the other team gains control.
  Loose balls, passes in flight, deflections, aerial duels and challenges keep the previous team.
  This is learned from the labels above, not hand-coded.
- **One-touch plays.** A single touch that goes straight back to the opponent or out of play does
  not create a possession.
- **Dead ball** runs from the ball crossing a line, or from the whistle (the players' collective
  deceleration), to the restart kick or throw. Stoppages under 1 s are not allowed.
- **Restarts.** The team taking the restart is whatever the network reads from the players and
  ball. The explicit restart-team rules of v1/v2 (last toucher's opponent, the tag's team, the
  nearest player) were measured and gave nothing on top of v3.
- **Before a kickoff.** Frames before a half's detected kickoff are `D`. A long wait before a
  kickoff in which the players never reach live-play speed is pinned to `D` whatever the networks
  say. The rule exists because BAR–ATH's second half is graded about 117 s before its kickoff,
  while players pass the ball around. None of the training matches holds a wait like that, so the
  networks called 63% of it live. The rule changes no held-out score (Metrica 0.8533 → 0.8533,
  DFL 0.7792 → 0.7792) and removes 5 spurious possessions from BAR–ATH.
- **`X`** is never written on a graded frame; ungraded frames are `X`.
- **No frame-specific constants.** Kickoffs, goals and stoppages are detected algorithmically.

## 6. Limitations

- **Short possessions are the main remaining error.** Held out, recall at IoU 0.5 is 0.08 for
  possessions under 1 s and 0.54 for 1–2 s, against 0.93–0.99 above 8 s. Most misses are never
  called for their team, or are swallowed by a longer neighbour.
- **Too few labelled games.** The networks fit their training games at 0.94–0.97 but score 0.85 held
  out, and training on one Metrica game instead of two costs 0.026. More hand-labelled other games
  (allowed by the rules) are the clearest next step.
- **BAR–ATH is a little harder than the held-out games.** All 8 networks agree on 83% of BAR–ATH's
  possessions, against 91% on held-out Metrica. Adding 0.1 m of extra noise to the players moves
  BAR–ATH's output more than Metrica's. BAR–ATH's own tracks are not that noisy, so we did not
  retrain for it, but it is a known weakness.
- **Dead ball with the ball in view.** v3 calls `D` on 14% of BAR–ATH's ball-seen frames, against
  about 20% expected from the tracking statistics, so it probably rides through some stoppages
  where players keep moving the ball. A stoppage prior from the HMM in the decoder was tried and cost
  0.01 on both validation sets.
- **Label convention uncertainty.** Our event-derived labels are not Metrica's
  corrected-tracking ownership. Under the alternative one-touch reading the held-out score is 0.721
  instead of 0.853, so the absolute number depends on where the organisers draw that line.
- **The networks are not retrained by our reproducibility check.** They were trained on a GPU,
  which is not bit-reproducible. The shipped JSON weights are the record, and everything from them
  to the CSV is reproduced byte for byte (§7).
- **Two hand-set numbers remain** in the kickoff-wait rule: the 10 s minimum and the 8 s speed
  window.

**With more time:** hand-label possessions on more open matches; train with tracking-noise
augmentation; run a per-input ablation with ≥ 4 training seeds per row.

## 7. Reproduction

Requirements: [uv](https://docs.astral.sh/uv/) ≥ 0.12 and Python 3.14 (uv installs it). The match
data bundle goes in `data/` as the organisers ship it, or set `MTAH_DATA_DIR`.

```bash
git clone https://github.com/Barca-Hackathon/more-than-a-hack-2027
cd more-than-a-hack-2027
# unpack the data bundle into data/  (data/tactical/BAR-ATH_tactical_atd.txt, ...)
uv sync --frozen
uv run more-than-a-hack-2027 predict --out evaluation/results/iRianEnjoyers.csv   # ~80 s, CPU
uv run python evaluation/validate_submission.py evaluation/results/iRianEnjoyers.csv
shasum -a 256 evaluation/results/iRianEnjoyers.csv
# expect 39e5382a8dd01638bab6ea9b34cc87c5240fa475d34754c3c0ec5970e6ea955a
```

`scripts/check_reproducibility.sh --head` runs the full check from a clean clone. It checks for data
leaks, predicts twice (cold and warm cache) and requires identical bytes, runs the organisers'
validator, retrains the control model from the public Metrica data and requires it to match the
shipped one byte for byte, and runs the test suite. The same hash was produced under 8 different
BLAS thread and kernel settings.

Other commands: `uv run pytest -q` runs the tests. `uv run more-than-a-hack-2027 eval-external`
scores methods on the external games. `uv run more-than-a-hack-2027 ablate` rebuilds the ablation
table. `uv run python scripts/kaggle_seq.py v3-final --push` retrains the shipped networks on a
Kaggle GPU.
