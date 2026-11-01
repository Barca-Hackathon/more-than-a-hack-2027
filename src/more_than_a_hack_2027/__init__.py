"""More than a Hack 2027, Phase 1: frame-level ball possession.

Layout:
    cli.py          the `more-than-a-hack-2027` command
    pipeline.py     the methods (v0 baseline ... v2): match -> frame labels
    submission.py   write the submission CSV and check it with the official validation
    _paths.py       repository locations; access to the organiser modules
    models/         shipped model weights (JSON)
    data/           the provided match: tracking, Smart Tagging, the shared Context
    stages/         pipeline building blocks: ball, nearest player, dead ball, control, restart,
                    decoder
    external/       other matches (Metrica) and the degradation that makes them look like
                    the competition data
    validation/     held-out scoring on the other matches: harness, decoder tuning, ablation
"""
