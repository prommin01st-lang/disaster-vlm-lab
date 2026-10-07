import pytest

from dvl.catalog import ALLOWED_TYPES
from dvl.mapping import crisismmd, disastervqa, pothole, traffic, wildfire


def test_crisismmd_not_informative_is_fixed_no_incident():
    assert crisismmd("hurricane_harvey", "not_informative", None) == (("no_incident",), "none")


def test_crisismmd_wildfire_candidates():
    cands, sev = crisismmd("california_wildfires", "informative", None)
    assert "forest_fire" in cands and "flood" not in cands and sev is None


def test_crisismmd_earthquake_maps_to_damage_not_earthquake():
    cands, _ = crisismmd("mexico_earthquake", "informative", "severe_damage")
    assert "earthquake" not in cands and "building_collapse" in cands


def test_crisismmd_damage_hint():
    assert crisismmd("srilanka_floods", "informative", "little_or_no_damage")[1] == "mild"
    assert crisismmd("srilanka_floods", "informative", "severe_damage")[1] == "severe"


def test_crisismmd_unknown_event_raises():
    with pytest.raises(KeyError):
        crisismmd("mars_quake", "informative", None)


@pytest.mark.parametrize("dt", ["flood", "accidents", "hurricane", "landslide", "fire", "storm",
                                "wildfire", "other_disasters", "other", "earthquake"])
def test_disastervqa_all_types_map_into_allowed(dt):
    cands = disastervqa(dt)
    assert cands and all(c in ALLOWED_TYPES for c in cands)


def test_wildfire_fixed():
    assert wildfire("nofire") == (("no_incident",), "none")
    assert wildfire("severe") == (("forest_fire",), "severe")
    assert wildfire("fire") == (("forest_fire",), None)


def test_traffic_and_pothole():
    assert traffic(True) == ("vehicle_collision",)
    assert traffic(False) == ("no_incident",)
    assert pothole("none") == (("no_incident",), "none")
    assert pothole("medium") == (("road_hazard",), "mild")
