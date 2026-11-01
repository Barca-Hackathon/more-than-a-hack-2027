"""Dead-ball stage on synthetic matches -- no gated data needed."""

import numpy as np

from more_than_a_hack_2027.stages import deadball
from more_than_a_hack_2027.stages.ball import clean_ball

CFG = deadball.DEADBALL_DEFAULTS


def test_touchline_out_is_detected_at_the_crossing(synthetic):
    n = 400
    ball = np.full((n, 2), np.nan)
    ball[:200, 0] = 50.0
    ball[:200, 1] = np.linspace(40.0, 75.0, 200)    # crosses y = 68 around frame 160
    ctx = synthetic(n, ball=ball)
    episodes = deadball.out_episodes(ctx, clean_ball(ctx), CFG)
    assert len(episodes) == 1
    crossing = int(np.argmax(ball[:, 1] > 68.0))
    assert abs(episodes[0].first - crossing) <= 1
    assert episodes[0].line == "touchline" and episodes[0].end == "high"


def test_line_noise_and_lofted_ball_are_not_outs(synthetic):
    n = 400
    ball = np.full((n, 2), 30.0)
    ball[:, 0] = 50.0
    ball[100:103, 1] = 68.2           # 0.2 m beyond for 3 frames: line noise
    ball[200:205, 1] = 71.0           # 3 m beyond for 5 frames, then straight back: lofted ball
    ball[205:, 1] = 66.0
    ctx = synthetic(n, ball=ball)
    assert deadball.out_episodes(ctx, clean_ball(ctx), CFG) == []


def test_kickoff_formation_and_kick_frame(synthetic):
    n = 300
    ctx = synthetic(n)
    xy = np.array(ctx.xy)
    xy[:100, 0, :] = (52.0, 34.0)       # a home kicker at the centre spot for 4 s ...
    xy[100:, 0, :] = (40.0, 34.0)       # ... then the formation breaks
    ctx = synthetic(n, xy=xy)
    formation, _, _ = deadball.kickoff_formation(ctx, CFG)
    kickoffs = deadball.detect_kickoffs(ctx, formation, CFG)
    assert len(kickoffs) == 1
    kick = kickoffs[0]
    assert kick.at_half_start and kick.kicker_team == 0
    assert abs(kick.frame - (100 - CFG["kickoff_lag_frames"])) <= 13   # formation smoothing


def test_kick_from_still_ball_is_a_restart_candidate(synthetic):
    n = 200
    ball = np.full((n, 2), np.nan)
    ball[:, 1] = 30.0
    ball[:100, 0] = 50.0
    ramp = np.cumsum(np.linspace(0.0, 0.6, 100))     # speed ramps 0 -> 15 m/s
    ball[100:, 0] = 50.0 + ramp
    ctx = synthetic(n, ball=ball)
    xy = np.array(ctx.xy)
    xy[:, 0, :] = (50.5, 30.0)                        # a player at the ball
    ctx = synthetic(n, ball=ball, xy=xy)
    track = clean_ball(ctx)
    distance = np.hypot(*(np.asarray(ctx.xy)[:, 0, :] - track.pos).T)
    events = deadball.candidate_restarts(ctx, track, distance, CFG)
    kicks = [f for f, kind in events if kind == "kick"]
    crossing = int(np.argmax(track.speed > CFG["kick_fast_mps"]))
    assert kicks and abs(kicks[0] - crossing) <= 1


def synthetic_match(make_ctx, segments, n_per_team=10, seed=0):
    """Live segments: fast, varying players and a moving visible ball. Dead: slow, steady, no ball."""
    rng = np.random.RandomState(seed)
    n = segments[-1][1]
    slots = 2 * n_per_team
    speed = np.zeros((n, slots), dtype=np.float32)
    ball = np.full((n, 2), np.nan)
    truth = np.zeros(n, dtype=bool)
    t = np.arange(n)
    for start, stop, dead in segments:
        part = slice(start, stop)
        if dead:
            truth[part] = True
            speed[part] = 1.0 + rng.normal(0, 0.05, size=(stop - start, slots))
        else:
            wave = 2.3 + 1.0 * np.sin(2 * np.pi * t[part] / 60.0)
            speed[part] = wave[:, None] + rng.normal(0, 0.3, size=(stop - start, slots))
            ball[part, 0] = 52 + 30 * np.sin(t[part] / 40.0)
            ball[part, 1] = 34 + 20 * np.cos(t[part] / 55.0)
    ctx = make_ctx(n, n_per_team=n_per_team, speed=np.clip(speed, 0, None), ball=ball)
    xy = np.array(ctx.xy)
    xy[:segments[0][1], 0, :] = (52.0, 34.0)          # kicker at the centre before kickoff
    return make_ctx(n, n_per_team=n_per_team, speed=np.clip(speed, 0, None), ball=ball, xy=xy), truth


def test_detect_recovers_stoppages_end_to_end(synthetic):
    segments = [(0, 60, True), (60, 1500, False), (1500, 1900, True), (1900, 3500, False),
                (3500, 3800, True), (3800, 6000, False), (6000, 6500, True), (6500, 8000, False)]
    ctx, truth = synthetic_match(synthetic, segments)
    result = deadball.detect(ctx)

    agreement = np.mean(result.mask == truth)
    assert agreement > 0.97, agreement
    assert len(result.stoppages) == 4
    starts = sorted(s.start for s in result.stoppages)[1:]
    for detected, (expected, _, _) in zip(starts, [s for s in segments[1:] if s[2]]):
        assert abs(detected - expected) <= 3 * ctx.fps
    assert result.params.mu[deadball.DEAD, 0] < result.params.mu[deadball.LIVE, 0]


def test_detect_is_deterministic(synthetic):
    segments = [(0, 60, True), (60, 1500, False), (1500, 1900, True), (1900, 3000, False)]
    ctx, _ = synthetic_match(synthetic, segments, seed=5)
    first, second = deadball.detect(ctx), deadball.detect(ctx)
    assert np.array_equal(first.mask, second.mask)
    assert np.array_equal(first.posterior, second.posterior, equal_nan=True)


def test_goal_anchor_needs_a_mid_half_kickoff_in_the_same_half(synthetic):
    ctx = synthetic(2000, halves=((1, 0, 999), (2, 1000, 1999)))
    wide = deadball.OutEpisode(900, 905, "goal_line", "high", True, 1.0)   # a shot just wide
    second_half = deadball.Kickoff(1100, 2, 1000, 1110, True, 1)
    anchors, confirmed = deadball.build_anchors(ctx, CFG, [wide], [second_half])
    assert not confirmed[0].goal
    hold = int(CFG["min_dead_after_out_s"] * ctx.fps)
    assert not anchors[900 + hold:1000].any()   # only the usual hold after an out is anchored

    restart = deadball.Kickoff(980, 1, 960, 990, False, 1)
    anchors, confirmed = deadball.build_anchors(ctx, CFG, [wide], [restart])
    assert confirmed[0].goal and anchors[900:980].all()


def test_late_first_kickoff_does_not_anchor_the_whole_stretch(synthetic):
    ctx = synthetic(2000, halves=((1, 0, 1999),))
    late = deadball.Kickoff(1500, 1, 1480, 1510, False, 0)    # the opening kickoff was missed
    cfg = dict(CFG, pre_kickoff_max_s=30.0)
    anchors, _ = deadball.build_anchors(ctx, cfg, [], [late])
    assert not anchors[:1480].any() and anchors[1480:1500].all()
    early = deadball.Kickoff(500, 1, 480, 510, False, 0)
    anchors, _ = deadball.build_anchors(ctx, cfg, [], [early])
    assert anchors[:500].all()
