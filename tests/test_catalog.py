from pathlib import Path

import pytest

from dvl.catalog import (ALLOWED_TYPES, PHASE1_TYPES, SEVERITIES, category_of,
                         sos_catalog_keys)

SOS_MD = Path("/home/petanque/Work/GitHubWork/SOS/docs/incident-catalog.md")


def test_phase1_has_12_types():
    assert len(PHASE1_TYPES) == 12


def test_extra_types():
    assert category_of("unsure") == "other"
    assert category_of("no_incident") is None


def test_category_of_known():
    assert category_of("flood") == "disaster"
    assert category_of("building_collapse") == "rescue"
    assert category_of("vehicle_collision") == "accident"


def test_category_of_unknown_raises():
    with pytest.raises(KeyError):
        category_of("earthquake")


def test_severities():
    assert SEVERITIES == ("none", "mild", "severe")


def test_allowed_is_phase1_plus_extra():
    assert set(ALLOWED_TYPES) == set(PHASE1_TYPES) | {"unsure", "no_incident"}


@pytest.mark.skipif(not SOS_MD.exists(), reason="SOS repo not present")
def test_phase1_matches_sos_catalog():
    sos = sos_catalog_keys(SOS_MD)
    assert len(sos) == 68
    for key, (cat, _) in PHASE1_TYPES.items():
        assert sos[key] == cat, key
    assert sos["unsure"] == "other"
