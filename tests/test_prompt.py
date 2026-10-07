from dvl.catalog import ALLOWED_TYPES
from dvl.prompt import (SYSTEM_PROMPT, USER_INSTRUCTION, build_messages,
                        build_teacher_messages)


def test_system_prompt_lists_every_allowed_type():
    for key in ALLOWED_TYPES:
        assert f"`{key}`" in SYSTEM_PROMPT, key


def test_system_prompt_is_stable():
    # ถ้าเทสต์นี้พัง = prompt เปลี่ยน → ต้องเทรนใหม่ทั้งหมด (อัปเดต hash ด้วยความตั้งใจเท่านั้น)
    import hashlib
    h = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12]
    assert h == SYSTEM_PROMPT_HASH


def test_build_messages_shape():
    msgs = build_messages("IMG")
    assert msgs[0] == {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"][0] == {"type": "image", "image": "IMG"}
    assert msgs[1]["content"][1] == {"type": "text", "text": USER_INSTRUCTION}


def test_build_messages_placeholder_for_training():
    assert build_messages(None)[1]["content"][0] == {"type": "image"}


def test_teacher_messages_restrict_to_candidates():
    msgs = build_teacher_messages("IMG", ("flood", "storm"))
    text = msgs[1]["content"][1]["text"]
    assert "`flood`" in text and "`storm`" in text
    assert "`forest_fire`" not in text
    assert "`no_incident`" in text and "`unsure`" in text  # ทางออกเสมอ


SYSTEM_PROMPT_HASH = "3456dbcbe0e7"
