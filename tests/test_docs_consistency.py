"""The docs a judge or the deck team reads must agree with the frozen config.

Catches the drift found on 2 Nov 2026: PPT docs still used the Rev A very-heavy threshold (124.5) and told
the deck team to say CHIRPS was the truth source after IMD became primary.
"""
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
DOCS = sorted([*ROOT.glob("*.md"), *(ROOT / "docs").glob("*.md"), ROOT / "mvp" / "README.md"])

# Lines that describe history on purpose (revision tables, or a line that also gives the corrected 115.6) may name the old value.
HISTORY = re.compile(r"115\.6|Rev A|was 124\.5|web said 124\.5|\| Thresholds \| 64\.5 / 124\.5")


def _lines(path: Path):
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        yield i, line


def test_config_thresholds_are_imd():
    assert CFG["exceed"]["thresholds"] == [64.5, 115.6]
    assert 64.5 in CFG["verify"]["thresholds"] and 115.6 in CFG["verify"]["thresholds"]


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_no_stale_very_heavy_threshold(path):
    bad = [f"{path.name}:{i}: {l.strip()[:90]}" for i, l in _lines(path) if "124.5" in l and not HISTORY.search(l)]
    assert not bad, "Rev A threshold 124.5 still used:\n" + "\n".join(bad)


def test_deck_brief_names_imd_as_truth():
    text = (ROOT / "docs" / "PPT_BRIEF.md").read_text(encoding="utf-8")
    assert "Say CHIRPS, not IMD" not in text
    assert "IMD 0.25° gridded rainfall" in text


def test_config_truth_matches_docs():
    assert CFG["sources"]["truth"] == "imd"


def test_pilot_numbers_in_readme_come_from_pilot_results():
    """Every decimal score quoted in README.md must appear in docs/PILOT_RESULTS.md (NFR-2)."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    pilot = (ROOT / "docs" / "PILOT_RESULTS.md").read_text(encoding="utf-8")
    for num in re.findall(r"\b0\.\d{3}\b", readme):
        assert num in pilot, f"README quotes {num}, which is not in PILOT_RESULTS.md"
