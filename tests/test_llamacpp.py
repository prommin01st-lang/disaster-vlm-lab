import base64, io, math

from PIL import Image

from dvl.llamacpp import (abstain, gate_fires, jpeg_data_url, p_incident, request_body, to_openai,
                          value_alternatives)
from dvl.prompt import build_messages


def test_to_openai_reads_path_and_passes_data_url(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNGfake")
    part = to_openai(build_messages(img))[1]["content"][0]
    assert part["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(b"\x89PNGfake").decode()
    url = "data:image/jpeg;base64,AAAA"
    part = to_openai(build_messages(url))[1]["content"][0]
    assert part == {"type": "image_url", "image_url": {"url": url}}


def test_request_body_uses_data_url():
    body = request_body(build_messages("data:image/jpeg;base64,AAAA"))
    assert body["messages"][1]["content"][0]["image_url"]["url"] == "data:image/jpeg;base64,AAAA"
    assert body["top_logprobs"] == 5


def test_jpeg_data_url_roundtrip():
    url = jpeg_data_url(Image.new("RGB", (40, 30), (200, 10, 10)))
    assert url.startswith("data:image/jpeg;base64,")
    img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert img.format == "JPEG" and img.size == (40, 30)


def test_value_alternatives_first_token_of_value():
    lp = [{"token": t, "logprob": -0.1, "top_logprobs": []} for t in ['{"', 'category', '":', 'null', ',"']]
    lp[3]["top_logprobs"] = [{"token": "null", "logprob": -0.01}, {"token": ' "', "logprob": -4.0}]
    assert value_alternatives(lp, "category") == [["null", -0.01], [' "', -4.0]]
    assert value_alternatives(lp, "severity") == []


def _pred(t, alts):
    return {"category": None, "incident_type": t, "severity": "none", "category_alts": alts}


def test_p_incident_and_gate():
    sure = _pred("no_incident", [["null", math.log(1 - 1e-6)], [' "', math.log(1e-6)]])
    shaky = _pred("no_incident", [["null", math.log(0.999)], [' "', math.log(0.001)]])
    assert math.isclose(p_incident(sure), 1e-6, rel_tol=1e-3)
    assert not gate_fires(sure, 3.2e-5)
    assert gate_fires(shaky, 3.2e-5)
    assert abstain(shaky, 3.2e-5)["incident_type"] == "unsure"
    assert abstain(shaky, 3.2e-5)["category"] == "other" and abstain(shaky, 3.2e-5)["severity"] == "mild"
    assert abstain(shaky, None) is shaky
    assert p_incident(_pred("no_incident", [])) == 1.0  # null หลุด top-5 → นับเป็นเหตุ
    fire = {**shaky, "incident_type": "forest_fire"}
    assert not gate_fires(fire, 3.2e-5)
