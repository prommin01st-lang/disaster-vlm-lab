"""label ของแต่ละ dataset → ชุด incident_type ที่เป็นไปได้ (teacher เลือกภายในชุดนี้)"""

_HURRICANE = ("storm", "flood", "damaged_structure", "building_collapse", "water_rescue", "road_hazard")
_CRISISMMD_EVENT = {
    "hurricane_harvey": _HURRICANE,
    "hurricane_irma": _HURRICANE,
    "hurricane_maria": _HURRICANE,
    "california_wildfires": ("forest_fire", "building_fire", "smoke_detected", "damaged_structure"),
    "mexico_earthquake": ("damaged_structure", "building_collapse", "landslide", "road_hazard"),
    "iraq_iran_earthquake": ("damaged_structure", "building_collapse", "landslide", "road_hazard"),
    "srilanka_floods": ("flood", "landslide", "water_rescue", "damaged_structure"),
}
_DAMAGE = {"little_or_no_damage": "mild", "mild_damage": "mild", "severe_damage": "severe"}


def crisismmd(event_name: str, informative: str, damage: str | None):
    cands = _CRISISMMD_EVENT[event_name]
    if informative == "not_informative":
        return ("no_incident",), "none"
    return cands, _DAMAGE.get(damage) if damage else None


_DVQA = {
    "flood": ("flood", "water_rescue", "damaged_structure"),
    "fire": ("building_fire", "vehicle_fire", "forest_fire", "smoke_detected"),
    "wildfire": ("forest_fire", "smoke_detected"),
    "landslide": ("landslide", "road_hazard"),
    "earthquake": ("damaged_structure", "building_collapse"),
    "hurricane": ("storm", "flood", "damaged_structure"),
    "storm": ("storm", "flood", "damaged_structure"),
    "accidents": ("vehicle_collision", "vehicle_fire"),
    "other_disasters": ("damaged_structure", "unsure"),
    "other": ("unsure",),
}


def disastervqa(disaster_type: str) -> tuple[str, ...]:
    return _DVQA[disaster_type]


_WILDFIRE_SEV = {"mild": "mild", "moderate": "severe", "severe": "severe", "fire": None}


def wildfire(label: str):
    if label == "nofire":
        return ("no_incident",), "none"
    return ("forest_fire",), _WILDFIRE_SEV[label]


def traffic(is_accident: bool) -> tuple[str, ...]:
    return ("vehicle_collision",) if is_accident else ("no_incident",)


_POTHOLE_SEV = {"low": "mild", "medium": "mild", "severe": "severe"}


def pothole(label: str):
    if label == "none":
        return ("no_incident",), "none"
    return ("road_hazard",), _POTHOLE_SEV[label]
