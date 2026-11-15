"""The submission writer on synthetic labels -- no match data needed."""

from more_than_a_hack_2027 import submission
from more_than_a_hack_2027.submission import pm


def test_write_csv_uses_lf_on_every_platform(tmp_path):
    path = tmp_path / "out" / "team.csv"
    submission.write_csv(str(path), ["X", "H", "A", "D"])
    assert path.read_bytes() == b"frame,label\n0,X\n1,H\n2,A\n3,D\n"


def test_write_csv_passes_the_official_reader(tmp_path):
    path = tmp_path / "team.csv"
    labels = ["X"] * 3 + ["H"] * 5 + ["D"] * 2
    submission.write_csv(str(path), labels)
    table = pm.read_label_csv(str(path))
    assert pm.validate_submission(table, len(labels)) is None
